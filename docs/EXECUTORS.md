# Task types and executor registry

The worker resolves `spec.taskType` against an executor registry in
`src/worker/runner.py`. Built-in executors:

| `taskType` | Executor | Use case |
|-----------|----------|----------|
| `echo` | `EchoExecutor` | Echo input back as result (smoke tests) |
| `inference` | `VLLMExecutor` / `TransformersExecutor` | LLM inference |
| `diffusion` | `DiffusersExecutor` | Image / video diffusion models |
| `omni_text2{audio,image,speech,general}` | `Omni*Executor` | Multimodal generation |
| `training` | `SFTExecutor` / `LoRASFTExecutor` / `DPOExecutor` / `PPOExecutor` | LLM fine-tuning |
| `image_classification_training` | `ImageClassificationTrainingExecutor` | Vision classification fine-tuning (`AutoModelForImageClassification` + HF `Trainer`) |
| `rag` | `RAGExecutor` | Retrieval-augmented generation |
| `agent` | `AgentExecutor` | Tool-using LLM agent (utu / youtu-agent backend) |
| `data_profiling` | `DataProfilingExecutor` | DataFrame profiling |
| `data_retrieval` | `DataRetrievalExecutor` | DataFrame loading from sources (`type: sql`, `type: s3`, `type: lumid` with `mode: sql\|s3\|agent` via lumid-data-app; `type: lumid` (mode `sql`/`s3`/`agent`) requires `lumid_data_token`, the bearer forwarded to lumid-data-app) |
| `ssh` | `SSHExecutor` | Interactive SSH session or non-interactive container job |

Helper utilities live in `src/worker/executors/utils/` (`artifacts`,
`checkpoints`, `data_utils`, `distributed`, `graph_templates`,
`huggingface`, `safe_eval`). Cross-cutting behavior is in
`src/worker/executors/mixins/` (`data`, `governance`, `inference`,
`training`).

## Result schema

Every executor's `run()` returns a subclass of `BaseExecutorResult`
(`src/shared/schemas/result.py`). The base class carries two
cross-cutting fields:

- `children: dict[str, BaseExecutorResult]` — per-child results when
  merged tasks share a dispatch.
- `artifacts: ArtifactContext | None` (wire key `_artifacts`) —
  resolution context for relative artifact refs.

Per-executor subclasses live next to the executor they describe — e.g.
`VLLMResult` in `src/worker/executors/vllm_executor.py`, `LoRAResult` in
`src/worker/executors/lora_sft_executor.py`. They add executor-specific
fields (`items`, `usage`, `final_lora`, `command`, …).

Artifact-bearing fields use `ArtifactRef` (`{"path": rel_path}`);
relative paths resolve against the producer's `_artifacts` context via
`artifact_to_source` / `_render_artifact_ref`.

## Agent executor (utu / youtu-agent)

`AgentExecutor` requires the following env vars to run; the executor
asserts them at import time, so a worker without them fails the task
immediately:

- `UTU_LLM_TYPE` — provider kind (e.g. `chat.completions`).
- `UTU_LLM_MODEL` — model identifier.
- `UTU_LLM_BASE_URL` — LLM endpoint base URL.
- `UTU_LLM_API_KEY` — LLM API key.

Optional, for the search tools:

- `SERPER_API_KEY`
- `JINA_API_KEY`
