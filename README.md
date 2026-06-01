# FlowMesh

[![arXiv](https://img.shields.io/badge/arXiv-2510.26913-b31b1b.svg)](https://arxiv.org/abs/2510.26913)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Lint](https://github.com/mlsys-io/FlowMesh/actions/workflows/lint-typecheck.yml/badge.svg)](https://github.com/mlsys-io/FlowMesh/actions/workflows/lint-typecheck.yml)
[![Tests](https://github.com/mlsys-io/FlowMesh/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/mlsys-io/FlowMesh/actions/workflows/unit-tests.yml)

A service fabric for running LLM agentic workflows on distributed GPU workers.

FlowMesh accepts workflow definitions (YAML, JSON, or n8n graph format), parses
them into a DAG of tasks, schedules and dispatches each task to a suitable
worker, and collects results and artifacts. It supports inference (vLLM, HF
transformers, diffusers), LLM training (SFT, LoRA, DPO, PPO), image
classification training, retrieval-augmented generation, agent execution,
SSH-style interactive sessions, and arbitrary container jobs — plus
miscellaneous data/utility tasks (data profiling, data retrieval, embeddings,
HTTP API calls, echo).

## Architecture

```
Client (CLI / SDK / HTTP)
    │
    ▼  HTTP (default 8000)
┌─────────────────────────────────────────────────┐
│ Server  (FastAPI orchestrator)                  │
│   • workflow parsing, DAG resolution            │
│   • task scheduling and dispatch                │
│   • result and artifact collection              │
│   • REST API + SSE log streaming                │
└──────────────┬──────────────────────────────────┘
               │  Redis pub/sub  (control + telemetry)
               ▼
┌─────────────────────────────────────────────────┐
│ Supervisor  (per-node agent)                    │
│   • registers node, manages worker containers   │
│   • relays tasks/events via gRPC streams        │
└──────────────┬──────────────────────────────────┘
               │  gRPC (default 50051)
               ▼
┌─────────────────────────────────────────────────┐
│ Worker  (executor process)                      │
│   • vllm, transformers, diffusers, training,    │
│     RAG, agent, SSH, echo, data profiling       │
│   • streams logs and events back via gRPC       │
└─────────────────────────────────────────────────┘
```

Server and Worker are the two top-level processes. The **Supervisor** is a
subsystem that lives under `src/server/` and runs as a child process spawned
from the server (`multiprocessing.Process`); single-node deployments spawn
one supervisor child alongside the server, multi-node deployments run a root
server plus one supervisor-only server process per worker node.

## Quick start

Requires Docker, Docker Compose, and Python 3.12+. If you want to use GPU
workers, ensure the NVIDIA Container Toolkit is also installed.

```bash
# 1. Install
git clone https://github.com/mlsys-io/FlowMesh.git
cd FlowMesh
pip install uv
uv sync --all-packages --group ci

# 2. Bring up the local stack (Server + Redis + Supervisor)
uv run flowmesh stack up

# 3. Start one CPU worker
uv run flowmesh stack worker up cpu 1

# 4. Submit a workflow
uv run flowmesh workflow submit examples/templates/echo_local.yaml

# 5. Watch it run
uv run flowmesh workflow list
uv run flowmesh workflow watch <workflow_id>
```

For a GPU worker:

```bash
# Pin to specific GPUs (or 'all')
uv run flowmesh stack worker up gpu --targets 0
```

For inference templates:

```bash
uv run flowmesh workflow submit examples/templates/inference_vllm_chat.yaml
uv run flowmesh workflow submit examples/templates/inference_hf_chat.yaml
```

Tear down:

```bash
uv run flowmesh stack worker down all
uv run flowmesh stack down
```

## Workflow format

A minimal single-task workflow:

```yaml
apiVersion: flowmesh/v1
kind: InferenceTask
metadata:
  name: hello-inference
spec:
  taskType: inference
  resources:
    hardware: { gpu: { type: any, count: 1 } }
  model:
    source: { type: huggingface, identifier: TinyLlama/TinyLlama-1.1B-Chat-v1.0 }
    vllm: { gpu_memory_utilization: 0.5 }
  data:
    type: list
    items:
      - - role: user
          content: What is the capital of France?
  inference: { max_tokens: 64, temperature: 0.0 }
  output:
    destination: { type: http }
```

Multi-stage DAGs, conditional execution, graph-template prompts, task merging,
and SSH sessions are all supported. See `examples/templates/` for end-to-end examples
and `AGENTS.md` for the full schema reference.

## Task types

Set `spec.taskType` to pick an executor. See [`docs/EXECUTORS.md`](docs/EXECUTORS.md)
for the full registry and per-executor fields.

| `taskType` | What it does |
|---|---|
| **LLM — inference & agents** | |
| `inference` | LLM inference (vLLM / HF transformers) |
| `omni_text2{audio,image,speech,general}` | Multimodal generation |
| `rag` | Retrieval-augmented generation |
| `agent` | Tool-using LLM agent |
| **LLM — training** | |
| `sft` / `lora_sft` / `dpo` / `ppo` | LLM fine-tuning (TRL) |
| **Deep learning (non-LLM)** | |
| `image_classification_training` | Vision classifier fine-tuning (HF `AutoModelForImageClassification`) |
| `diffusion` | Image / video diffusion models |
| **Misc / data & utility** | |
| `data_profiling` | DataFrame profiling |
| `data_retrieval` | DataFrame loading from sources |
| `embedding` | Text embeddings |
| `api` | Outbound HTTP API call step |
| `ssh` | Interactive SSH session or non-interactive container job |
| `echo` | Echo input back (smoke tests) |

## Extending FlowMesh

FlowMesh exposes plugin hooks for organisations that want to layer additional
auth, submission policy, usage tracking, authorisation, supplier attribution,
or resource lifecycle behaviour on top of the core server. Install the
standalone hook contract with:

```bash
pip install "flowmesh[hook]"
```

A plugin is any Python module that exposes `install()` returning
`flowmesh_hook.HookBindings`. Plugins are loaded by setting
`FLOWMESH_PLUGINS` to a comma-separated list of importable module names.
Plugins can ship as in-tree modules, sibling-mounted packages, or
pip-installable wheels — the core never references plugin names.

See [`docs/PLUGINS.md`](docs/PLUGINS.md) for the full plugin contract.

## Development

```bash
# Install dev tooling
uv sync --all-packages --group ci

# Format / lint / type-check
uv run pre-commit run --all-files

# Tests — skip the multiprocessing GPU-cleanup test because it requires a
# real CUDA device and isolated processes; CI also skips it.
uv run pytest tests/ --ignore=tests/worker/test_mp_executor_cleanup_gpu.py
```

Detailed contributor docs (project layout, env vars, dispatch internals,
executor registry, commit-message conventions) live in `AGENTS.md`.

## Contributing

We welcome bug fixes, new features, documentation improvements, and feedback.
Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contributor setup,
code style, testing, dependency-pin, and DCO sign-off conventions, and
[`AGENTS.md`](AGENTS.md) for a deeper architecture and source-layout tour.

## Citation

If you use FlowMesh in your research, please cite:

```bibtex
@misc{shen2025flowmesh,
      title={FlowMesh: A Service Fabric for Composable LLM Workflows}, 
      author={Junyi Shen and Noppanat Wadlom and Lingfeng Zhou and Dequan Wang and Xu Miao and Lei Fang and Yao Lu},
      year={2025},
      eprint={2510.26913},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2510.26913}, 
}
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
