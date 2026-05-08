<!-- markdownlint-disable -->

<!--
PR title format (enforced by CI):
  type: description
  e.g. feat: add retry logic for task dispatch
       fix: handle OOM in vLLM executor
       [BREAKING] refactor: rename FlowMesh client methods

Types: feat, fix, refactor, chore, test, perf
-->

## Purpose

<!-- What does this PR do? Reference related issues with "Fixes #123" or "Relates to #123". -->

## Changes

<!-- List modified files or groups of files with a brief explanation of each change. -->
<!--
- `src/server/dispatcher/base.py` — added retry logic for failed dispatches
- `src/server/dispatcher/factory.py` — wire up new retry config
- `tests/server/dispatcher/test_dispatcher.py` — cover retry and backoff cases
-->

## Design

<!-- For non-trivial PRs: explain the high-level approach and any alternatives you considered. -->

## Test Plan

<!-- How were these changes validated? Provide commands, sample workflows, or screenshots. -->

## Test Result

<!-- Paste relevant test output, logs, or before/after comparisons. -->

---

<details>
<summary>Pre-submission Checklist</summary>

- [ ] I have read the contribution guidelines.
- [ ] I have run `pre-commit run --all-files` and fixed any issues.
- [ ] I have added or updated tests covering my changes (if applicable).
- [ ] I have verified that `uv run pytest tests/` passes locally.
- [ ] If I changed shared schemas or proto definitions, I have checked downstream compatibility across Server and Worker.
- [ ] If I changed the SDK or CLI, I have verified the affected packages work (`uv sync --all-packages --group ci --frozen`).
- [ ] If this is a breaking change, I have prefixed the PR title with `[BREAKING]` and described migration steps above.
- [ ] I have updated documentation or config examples if user-facing behavior changed.

</details>
