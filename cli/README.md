# FlowMesh CLI

Standalone FlowMesh CLI package (`flowmesh-cli`).

## Packages

- `flowmesh-cli` (base): core commands — workflow, task, worker, node, ssh,
  result, system, plus basic `info` / `health` helpers.
- `flowmesh-cli-stack` (`cli/stack/`): stack deployment commands — build, up,
  down, worker management, bundle, env helpers, doctor.

Install with the `stack` extra to get both:

```bash
pip install "flowmesh[cli]"
```

The CLI entrypoint is `flowmesh`. Stack assets (compose, bake, env example) are
bundled under `flowmesh_cli_stack/assets/`.
