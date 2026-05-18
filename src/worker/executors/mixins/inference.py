import copy
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from PIL import Image

from shared.schemas.governance import SpanType
from shared.tasks.specs import InferenceSpecStrict
from shared.utils.json import to_json_serializable

from ..base_executor import ExecutionError
from .data import DataMixin, InferenceEntry, PromptInput

logger = logging.getLogger(__name__)

_RESERVED_CHAT_TEMPLATE_KWARGS = frozenset(
    {"messages", "tokenize", "add_generation_prompt"}
)

type RenderedChatMessage = dict[str, Any]
type MetadataPrompt = str | Sequence[RenderedChatMessage]
type PromptMetadata = dict[str, Any]


@dataclass(slots=True)
class PreparedInferenceEntry:
    task_id: str
    prompts: list[str]
    inference_cfg: dict[str, Any]
    data_cfg: dict[str, Any]
    metadata: list[PromptMetadata]
    images: list[Image.Image | None]
    image_group_sizes: list[int] | None
    image_embedding_path: Path | None
    tables: list[pd.DataFrame]
    applied_chat_template: bool

    image_embedding: torch.Tensor | None = None
    image_group_base_prompts: list[str] | None = None
    image_group_base_metadata: list[PromptMetadata] | None = None


class InferenceMixin(DataMixin):
    """Shared helpers for inference-style executors."""

    def _get_tokenizer(self) -> Any | None:
        return None

    def _should_apply_chat_template(self) -> bool:
        tokenizer = self._get_tokenizer()
        return tokenizer is not None and (
            getattr(tokenizer, "chat_template", None) is not None
        )

    def _apply_chat_template(
        self,
        prompts: Sequence[PromptInput],
        system_prompt: str | None,
        has_images: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[list[RenderedChatMessage]], list[str]]:
        """Render prompts via ``tokenizer.apply_chat_template``. Extra
        ``chat_template_kwargs`` (e.g. Qwen3 ``enable_thinking``) are forwarded."""
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            raise ExecutionError(
                "Tokenizer is not available for applying chat template"
            )
        if getattr(tokenizer, "chat_template", None) is None:
            raise ExecutionError("Chat template is not available in the tokenizer")
        extra_kwargs: dict[str, Any] = chat_template_kwargs or {}
        structured_prompts: list[list[RenderedChatMessage]] = []
        formatted_prompts: list[str] = []
        for prompt in prompts:
            messages: list[RenderedChatMessage] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if isinstance(prompt, str):
                if has_images:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    )
                else:
                    messages.append({"role": "user", "content": prompt})
            else:
                if not (
                    isinstance(prompt, list)
                    and all(isinstance(m, dict) for m in prompt)
                ):
                    raise ExecutionError(
                        f"All messages in prompt must be dicts but got: {prompt!r}"
                    )
                if prompt[-1].get("role") != "user":
                    raise ExecutionError(
                        "Last message in prompt must have role 'user' for generation"
                    )
                formatted_prompt: list[RenderedChatMessage] = [dict(m) for m in prompt]
                if has_images:
                    formatted_prompt[-1] = {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt[-1]["content"]},
                        ],
                    }
                messages.extend(formatted_prompt)
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **extra_kwargs,
            )
            structured_prompts.append(messages)
            formatted_prompts.append(formatted)

        return structured_prompts, formatted_prompts

    def _normalize_inference_for_sampling(
        self, inference: dict[str, Any]
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in inference.items():
            if key in {"system_prompt", "apply_chat_template"}:
                continue
            normalized[key] = copy.deepcopy(value)
        return normalized

    def _normalize_metadata_row(
        self, meta: dict | None, base_prompt: MetadataPrompt
    ) -> PromptMetadata:
        normalized: PromptMetadata = {
            "prompt": base_prompt,
        }
        if meta is None:
            return normalized
        if not isinstance(meta, dict):
            raise ExecutionError("spec.data.metadata entries must be dictionaries")
        for key, value in meta.items():
            normalized[str(key)] = value
        return normalized

    def _build_metadata_rows(
        self, metadata_raw: list[Any], prompts: Sequence[MetadataPrompt]
    ) -> list[PromptMetadata]:
        metadata_rows: list[PromptMetadata] = []
        if metadata_raw:
            if len(metadata_raw) != len(prompts):
                raise ExecutionError(
                    "spec.data.metadata length must match number of prompts"
                )
            for meta, metadata_prompt in zip(metadata_raw, prompts):
                metadata_rows.append(
                    self._normalize_metadata_row(meta, metadata_prompt)
                )
            return metadata_rows

        for metadata_prompt in prompts:
            metadata_rows.append(self._normalize_metadata_row({}, metadata_prompt))
        return metadata_rows

    @staticmethod
    def _field_has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) > 0
        return True

    def _resolve_export_field(
        self, cfg: Any, *, item: dict[str, Any], prompt: str, metadata: dict[str, Any]
    ) -> Any:
        key: Any
        if isinstance(cfg, str):
            if cfg == "prompt":
                return prompt
            if cfg == "output":
                return item.get("output")
            if cfg.startswith("metadata."):
                key = cfg.split(".", 1)[1]
                return metadata.get(key)
            if cfg.startswith("item."):
                key = cfg.split(".", 1)[1]
                return item.get(key)
            return cfg

        if not isinstance(cfg, dict):
            return cfg

        source = cfg.get("from") or cfg.get("source")
        if source in {None, "literal"}:
            value = cfg.get("value", cfg.get("literal"))
        elif source == "prompt":
            value = prompt
        elif source == "output":
            value = item.get("output")
        elif source == "metadata":
            key = cfg.get("key")
            value = metadata.get(key) if key else metadata
        elif source == "item":
            key = cfg.get("key")
            value = item.get(key) if key else item
        else:
            value = None

        if (
            value is None or (isinstance(value, str) and value.strip() == "")
        ) and "fallback" in cfg:
            value = cfg.get("fallback")
        if value is None and "default" in cfg:
            value = cfg.get("default")
        return value

    def _maybe_export_jsonl(
        self,
        spec: InferenceSpecStrict,
        task_id: str,
        result: dict[str, Any],
        out_dir: Path,
    ) -> None:
        post_cfg = (postprocess := spec.postprocess) and postprocess.jsonl_export
        if not post_cfg:
            return

        path = post_cfg.path
        fields_cfg = post_cfg.fields
        if not fields_cfg:
            raise ExecutionError(
                "postprocess.jsonl_export.fields must define at least one field"
            )

        artifacts_dir = (out_dir / "artifacts").resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        target_rel = Path(path)
        if target_rel.is_absolute():
            raise ExecutionError(
                "postprocess.jsonl_export.path must be relative to artifacts/ "
                f"(got {path})"
            )
        target_path = (artifacts_dir / target_rel).resolve()
        try:
            target_path.relative_to(artifacts_dir)
        except ValueError as exc:
            raise ExecutionError(
                "postprocess.jsonl_export.path must stay under artifacts/ "
                f"(got {path})"
            ) from exc
        target_path.parent.mkdir(parents=True, exist_ok=True)

        items = result.get("items") or []
        required_fields = post_cfg.required_fields or []
        records: list[dict[str, Any]] = []

        for idx, item in enumerate(items):
            prompt_text = item.get("prompt") or ""
            metadata = item.get("metadata") or {}
            record: dict[str, Any] = {}
            for field_name, field_cfg in fields_cfg.items():
                value = self._resolve_export_field(
                    field_cfg, item=item, prompt=prompt_text, metadata=metadata
                )
                record[field_name] = to_json_serializable(value)
            missing = [
                name
                for name in required_fields
                if not self._field_has_value(record.get(name))
            ]
            if missing:
                raise ExecutionError(
                    f"postprocess.jsonl_export missing required fields {missing} for "
                    f"item {idx} (task {task_id})"
                )
            records.append(record)

        with target_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False))
                fh.write("\n")

        rel_path = target_path.relative_to(artifacts_dir).as_posix()
        logger.info(
            "Task %s exported %d records to artifacts/%s",
            task_id,
            len(records),
            rel_path,
        )

    @staticmethod
    def _extract_chat_template_kwargs(
        inference_cfg: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Validate ``inference.chat_template_kwargs``; reject non-mappings
        and overrides of worker-controlled keys."""
        raw = inference_cfg.get("chat_template_kwargs")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ExecutionError(
                "spec.inference.chat_template_kwargs must be a mapping of "
                "Jinja context variables (e.g. {'enable_thinking': False})."
            )
        clash = sorted(set(raw) & _RESERVED_CHAT_TEMPLATE_KWARGS)
        if clash:
            raise ExecutionError(
                "spec.inference.chat_template_kwargs must not override "
                f"worker-controlled arguments: {clash}. These keys are set "
                "by the executor itself."
            )
        return raw

    def _prepare_inference_entry(
        self, entry: InferenceEntry, *, has_images: bool = False
    ) -> PreparedInferenceEntry:
        task_id = entry.task_id
        with self._span(
            "prompt postprocessing",
            span_type=SpanType.COMPUTE,
            data_id=task_id,
        ):
            inference_cfg = entry.inference_cfg
            append_system_prompt = entry.append_system_prompt
            system_prompt = inference_cfg.get("system_prompt")
            metadata_raw = entry.metadata_raw
            prompts = entry.prompts

            apply_chat_template = bool(
                inference_cfg.get(
                    "apply_chat_template", self._should_apply_chat_template()
                )
            )
            chat_template_kwargs = self._extract_chat_template_kwargs(inference_cfg)
            metadata_prompts: Sequence[MetadataPrompt]

            if apply_chat_template:
                metadata_prompts, rendered_prompts = self._apply_chat_template(
                    prompts,
                    system_prompt if append_system_prompt else None,
                    has_images=has_images,
                    chat_template_kwargs=chat_template_kwargs,
                )
            else:
                if prompts and not isinstance(prompts[0], str):
                    raise ExecutionError(
                        "Chat-style prompts require apply_chat_template=true and a "
                        "tokenizer with a chat template."
                    )
                prompts_as_text = cast(list[str], prompts)
                metadata_prompts = prompts_as_text
                if system_prompt and append_system_prompt:
                    rendered_prompts = [
                        f"{system_prompt}\n{prompt}" for prompt in prompts_as_text
                    ]
                else:
                    rendered_prompts = prompts_as_text.copy()

            metadata_rows = self._build_metadata_rows(metadata_raw, metadata_prompts)

            return PreparedInferenceEntry(
                task_id=task_id,
                prompts=rendered_prompts,
                inference_cfg=inference_cfg,
                data_cfg=entry.data_cfg,
                metadata=metadata_rows,
                images=entry.images,
                image_group_sizes=entry.image_group_sizes,
                image_embedding_path=entry.image_embedding_path,
                tables=entry.tables,
                applied_chat_template=apply_chat_template,
            )
