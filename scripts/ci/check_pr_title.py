"""Validate PR title format: [BREAKING] type: description."""

import os
import re
import sys

pr_title = os.environ.get("PR_TITLE", "").strip()
if not pr_title:
    print("❌ PR title is empty.")
    sys.exit(1)

ALLOWED_TYPES = ["feat", "fix", "refactor", "chore", "test", "perf", "docs"]

# Strip optional [1/N] progress prefix
progress_match = re.match(r"^\[\d+/[\dNn]+\]\s*(.+)$", pr_title, re.IGNORECASE)
if progress_match:
    pr_title = progress_match.group(1).strip()

# Strip optional [BREAKING] prefix
breaking_match = re.match(r"^\[BREAKING\]\s*(.+)$", pr_title, re.IGNORECASE)
if breaking_match:
    core_title = breaking_match.group(1).strip()
    is_breaking = True
else:
    core_title = pr_title
    is_breaking = False

# Validate type: feat, fix, refactor, chore, test, perf, docs
types_re = "|".join(re.escape(t) for t in ALLOWED_TYPES)
type_match = re.match(rf"^({types_re}):\s+.+$", core_title, re.IGNORECASE)
if not type_match:
    print(f"❌ Invalid PR title: '{pr_title}'")
    print("   Expected format: type: description")
    print(f"   Allowed types: {', '.join(ALLOWED_TYPES)}")
    sys.exit(1)

change_type = type_match.group(1).lower()
breaking_info = " (BREAKING CHANGE)" if is_breaking else ""
print(f"✅ PR title is valid: {pr_title}\n" f"   type: {change_type}{breaking_info}")
