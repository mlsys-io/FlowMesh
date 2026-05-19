#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation - Retrieval Only) Executor

This executor queries a Qdrant collection using server-side embeddings.
Supports single or multiple queries.
"""

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from datasets import load_dataset
from qdrant_client import QdrantClient, models

from shared.schemas.result import BaseExecutorResult
from shared.tasks.specs import RagSpecStrict

from .base_executor import ExecutionError, Executor, ExecutorTask
from .utils.graph_templates import Message, build_prompts_from_graph_template

logger = logging.getLogger("worker.rag")


class RAGResult(BaseExecutorResult):
    ok: bool = True
    executor: str
    qdrant: dict[str, Any]
    embedding: dict[str, Any]
    search: dict[str, Any]
    queries: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None


class RAGExecutor(Executor):
    name = "rag"

    def run(self, task: ExecutorTask, out_dir: Path) -> RAGResult:
        start_ts = time.time()
        spec = self.require_spec(task, RagSpecStrict)

        qdrant_cfg = spec.qdrant or {}
        url = qdrant_cfg.get("url")
        api_key = qdrant_cfg.get("api_key")
        collection = qdrant_cfg.get("collection")

        embedding_cfg = spec.embedding or {}
        model_name = embedding_cfg.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )

        search_cfg = spec.search or {}
        top_k = int(search_cfg.get("top_k", 5))

        # Prepare queries (dataset | list | graph_template | single query)
        queries: Sequence[str | dict[str, Any] | Message] = []
        data_cfg = spec.data or {}
        dtype = data_cfg.get("type") if isinstance(data_cfg, dict) else None
        if dtype == "dataset":
            data_url = data_cfg.get("url")
            if not data_url:
                raise ExecutionError("spec.data.url is required for type == 'dataset'.")
            name = data_cfg.get("name", None)
            split = data_cfg.get("split", "train")
            shuffle = bool(data_cfg.get("shuffle", False))
            column = data_cfg.get("column", "text")

            trust_remote_code = data_cfg.get("trust_remote_code")
            revision = data_cfg.get("revision")
            dataset_kwargs = {
                "name": name,
                "split": split,
                "revision": revision,
            }
            if trust_remote_code is not None:
                dataset_kwargs["trust_remote_code"] = bool(trust_remote_code)
            dataset = load_dataset(
                data_url,
                **{k: v for k, v in dataset_kwargs.items() if v is not None},
            )
            if shuffle:
                seed = int(data_cfg.get("seed", 42))
                buffer_size = data_cfg.get("buffer_size", None)
                dataset = (
                    dataset.shuffle(seed=seed)
                    if buffer_size is None
                    else dataset.shuffle(
                        seed=seed,
                        buffer_size=int(buffer_size),  # type: ignore[call-arg]
                    )
                )

            if column not in dataset.column_names:
                raise ExecutionError(
                    f"Column '{column}' not found in dataset. "
                    f"Available: {dataset.column_names}"
                )
            queries = [str(x) for x in dataset[column]]
        elif dtype == "list":
            item_list = data_cfg.get("items", [])
            if not isinstance(item_list, list) or any(
                not isinstance(x, str) for x in item_list
            ):
                raise ExecutionError(
                    "spec.data.items must be a list of strings for type == 'list'."
                )
            queries = [s for s in item_list]
        elif dtype == "graph_template":
            # Build queries from upstream results using the graph template
            queries = build_prompts_from_graph_template(data_cfg, spec)
        else:
            # Backward compatibility: spec.query as a single string
            query_text = spec.query
            if query_text is not None and query_text.strip():
                queries = [query_text]
            else:
                raise ExecutionError(
                    "Missing input queries: provide spec.query or spec.data"
                )

        # Basic validation
        if not url:
            raise ExecutionError("Missing spec.qdrant.url")
        if not collection:
            raise ExecutionError("Missing spec.qdrant.collection")
        if not queries:
            raise ExecutionError(
                "No queries prepared. Check spec.query or spec.data configuration."
            )

        logger.info("Connecting Qdrant url=%s collection=%s", url, collection)
        client = (
            QdrantClient(url=url, api_key=api_key) if api_key else QdrantClient(url=url)
        )

        results_per_query: list[dict[str, Any]] = []
        total_items = 0
        for i, q in enumerate(queries):
            try:
                logger.info("Querying top_k=%d using model=%s", top_k, model_name)
                res = client.query_points(
                    collection_name=collection,
                    query=models.Document(text=str(q), model=model_name),
                    limit=top_k,
                )
                points = getattr(res, "points", []) or []
            except Exception as e:
                has_api_key = bool(api_key)
                scheme = host = ""
                try:
                    parsed = urlparse(url or "")
                    scheme = parsed.scheme
                    host = parsed.netloc
                except Exception:
                    pass
                err_msg = f"Qdrant query failed: {e}"
                ctx_msg = (
                    f"url={url} (scheme={scheme}, host={host}), "
                    f"collection={collection}, has_api_key={has_api_key}"
                )
                logger.error(
                    "%s; context: %s; exception_type=%s",
                    err_msg,
                    ctx_msg,
                    type(e).__name__,
                )
                print(
                    f"[RAGExecutor] {err_msg}; context: {ctx_msg}; "
                    f"exception_type={type(e).__name__}"
                )
                raise ExecutionError(f"{err_msg}. {ctx_msg}")

            items: list[dict[str, Any]] = []
            for p in points:
                items.append(
                    {
                        "id": getattr(p, "id", None),
                        "score": getattr(p, "score", None),
                        "payload": getattr(p, "payload", None),
                    }
                )
            total_items += len(items)
            results_per_query.append(
                {
                    "index": i,
                    "query": str(q),
                    "items": items,
                }
            )

        logger.info(
            "RAG query completed queries=%d total_results=%d", len(queries), total_items
        )
        return RAGResult(
            ok=True,
            executor=self.name,
            qdrant={"collection": collection, "url": url},
            embedding={"model": model_name},
            search={"top_k": top_k},
            queries=results_per_query,
            usage={
                "latency_sec": round(time.time() - start_ts, 4),
                "num_queries": len(queries),
                "total_results": total_items,
            },
        )
