# Workflow YAML format

Workflows are submitted as YAML (or JSON) to `POST /api/v1/workflows`
(see [`docs/API.md`](API.md)). The `templates/` directory contains
runnable examples for each shape; this page documents the spec
hierarchy and the cross-cutting features.

## Single task

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

## Multi-stage DAG

```yaml
apiVersion: flowmesh/v1
kind: Workflow
spec:
  stages:
    - name: extract
      spec:
        taskType: inference
        ...
        data:
          type: list
          items:
            - - role: user
                content: "Extract entities from: {{input}}"
    - name: summarize
      dependsOn: [extract]
      spec:
        taskType: inference
        ...
        data:
          type: list
          items:
            - - role: user
                content: "Summarize: {{extract.output}}"
```

`spec.stages[].dependsOn` declares the DAG edges; the dispatcher
schedules each stage once all of its dependencies are `DONE`.
Substitutions like `{{extract.output}}` are resolved against the
upstream stage's result.

## Graph DAG

`taskType: graph_template` — topology-aware multi-input prompts with
parent output substitution and validation. See
`src/worker/executors/utils/graph_templates.py` for the templating
contract.

## Schedule hints

Workflows can declare scheduling preferences via
`metadata.annotations.schedule_hint`:

- `epoch_groups: [[<task_name>, ...], ...]` — epoch-ordered execution;
  tasks in epoch `n` only dispatch after every task in epoch `n-1`
  succeeds.
- `schedule_in_epoch_order: true` — for dependent DAGs, prefer
  position-in-epoch tie-breaks during dispatch.
