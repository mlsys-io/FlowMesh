# Task types and executor registry

The worker resolves `spec.taskType` against an executor registry in
`src/worker/runner.py`. Built-in executors:

| `taskType` | Executor | Use case |
|-----------|----------|----------|
| `echo` | `EchoExecutor` | Echo input back as result (smoke tests) |
| `inference` | `VLLMExecutor` / `TransformersExecutor` | LLM inference |
| `embedding` | `VLLMEmbeddingExecutor` (text, when `model.vllm` is set) / `TransformersExecutor` (visual, `model.transformers.mode: visual-embedding`) | Text / visual embeddings |
| `diffusion` | `DiffusersExecutor` | Image / video diffusion models |
| `omni_text2{audio,image,speech,general}` | `Omni*Executor` | Multimodal generation |
| `training` | `SFTExecutor` / `LoRASFTExecutor` / `DPOExecutor` / `PPOExecutor` | LLM fine-tuning |
| `image_classification_training` | `ImageClassificationTrainingExecutor` | Vision classification fine-tuning (`AutoModelForImageClassification` + HF `Trainer`) |
| `rag` | `RAGExecutor` | Retrieval-augmented generation |
| `agent` | `AgentExecutor` | Tool-using LLM agent (utu / youtu-agent backend) |
| `data_profiling` | `DataProfilingExecutor` | DataFrame profiling |
| `data_retrieval` | `DataRetrievalExecutor` | DataFrame loading from sources (`type: sql`, `type: s3`, `type: lumid` with `mode: sql\|s3\|agent` via lumid-data-app; `type: lumid` (mode `sql`/`s3`/`agent`) requires `lumid_data_token`, the bearer forwarded to lumid-data-app) |
| `ssh` | `SSHExecutor` | Interactive SSH session or non-interactive container job |
| `python` | `PythonExecutor` | A user Python function run as a workflow stage in an isolated container (Docker session backend only) |
| `api` | `APIExecutor` | One parallel HTTP request per `spec.data` row |
| `serve` | `VLLMServeExecutor` | Persistent vLLM API server for a single model |

Helper utilities live in `src/worker/executors/utils/` (`artifacts`,
`checkpoints`, `data_utils`, `distributed`, `graph_templates`,
`huggingface`, `safe_eval`). Cross-cutting behavior is in
`src/worker/executors/mixins/` (`data`, `governance`, `inference`,
`training`).

## Result schema

Every executor's `run()` returns an exact per-task-type subclass of
`BaseExecutorResult`, all defined in the shared `src/shared/schemas/result`
package. The base class carries two cross-cutting fields:

- `children: dict[str, BaseExecutorResult]` — per-child results when
  merged tasks share a dispatch.
- `artifacts: ArtifactContext | None` (wire key `_artifacts`) —
  resolution context for relative artifact refs.

Each subclass declares its exact fields (typed nested payloads) and tags itself
with a `task_type` discriminator — e.g. `InferenceResult`, `LoRAResult`,
`AgentResult`, `SSHResult`. The `AnyExecutorResult` discriminated union in
the same package deserializes a `results.json` back into its exact subclass
end-to-end (worker envelope → server ingest and `GET /results/{id}` → SDK).
Results without a `task_type` (legacy files, condition-skips) fall back to
the permissive base.

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

## SSH executor (process backend)

On a root worker, a `process` session runs under its own account, which is
denied the worker's state through a POSIX ACL entry on each of: `RESULTS_DIR`,
the directory holding `WORKER_HB_FILE`, the worker's home, any of `HF_HOME`,
`HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_DATASETS_CACHE`,
`TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT` and
`FASTEMBED_CACHE_PATH` that is set, and the `utu` and `fastembed_cache`
directories in the temp dir. The worker therefore needs the `acl` package
(`setfacl` / `getfacl`) and ACL support on the filesystems behind those paths;
without either, it does not offer `process`. It also does not offer `process`
when:

- another worker in the same container, or on the same machine outside
  containers, already serves `process` sessions;
- one of those paths contains a directory every session needs, such as the
  temp dir or `/mnt/flowmesh`;
- a directory that resolving one of those paths passes through, links
  included, is world-writable, unless it is one of those paths or inside one,
  or it is sticky and holds the next component as a directory the worker owns
  rather than a link.

Each session account has a group of its own, and takes its uid and gid from
61000–64999. The deny entries do not cover files an agent tool writes directly
into the temp dir.

A session's inputs and output live in its own directory in the worker's temp
dir (`TMPDIR`, `/tmp` by default), so that filesystem must have room for them.
Each `mountPath` is a link to them under `/mnt/flowmesh`. `/mnt/flowmesh` is
emptied before and after every session, so it must not have a filesystem
mounted below it or be shared between workers (for example, one host directory
bind-mounted into several containers). A `mountPath` must name a path below
`/mnt/flowmesh`, must not contain `..`, must have at most 32 components and 1024
characters, and must not be nested inside another one. Output is collected as
the regular files the session owns; links and special files are dropped, and
output nested more than 64 directories deep fails the task.
