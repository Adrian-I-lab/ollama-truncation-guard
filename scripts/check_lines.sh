#!/usr/bin/env bash
# Gate for the promise of fewer than 200 lines of Python, excluding tests.
# Counts PHYSICAL lines (wc -l), so blank lines and comments count too.
# Run from the repo root. Override the limit with MAX=<n>.
set -euo pipefail
MAX="${MAX:-199}"

files=$(find . -name '*.py' -not -path './tests/*' -not -path './.git/*' -not -path './.venv/*' | sort)
count=$(printf '%s' "$files" | grep -c . || true)
echo "counted files:"
printf '%s\n' "$files" | sed 's/^/  /'
if [ "$count" -eq 0 ]; then
  echo "python_lines_excluding_tests=0 max=$MAX files=0 (nothing counted, failing)"
  exit 1
fi
total=$(find . -name '*.py' -not -path './tests/*' -not -path './.git/*' -not -path './.venv/*' -print0 \
  | xargs -0 cat | wc -l | tr -d ' ')
echo "python_lines_excluding_tests=$total max=$MAX files=$count"
[ "$total" -le "$MAX" ]
