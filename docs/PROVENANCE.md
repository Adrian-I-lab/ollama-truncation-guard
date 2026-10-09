# Provenance

Every file in this repository was written from scratch on 2026-10-09, the date range of its commits. It was written against the documented public behaviour of the Ollama API. That means the `prompt_eval_count` field, the `options.num_ctx` setting, and newline-delimited JSON streaming.

No code, data, or text was copied from any other codebase.

The motivating observation is described generically in the README. A prompt was silently cut to about half the context window. That is an observation, not data shipped in this repository.

The tests use a synthetic fake server. No real model output is stored here.

## Files by category

- **Logic.** `guard.py`.
- **Integration.** `proxy.py`.
- **Test doubles and tests.** `tests/fake_ollama.py`, `tests/test_guard.py`, `tests/test_fake.py`, `tests/test_proxy.py`, and `tests/test_real_ollama.py`.
- **Scripts.** `scripts/check_lines.sh` and the git hooks in `scripts/hooks/`.
- **Docs.** `README.md`, `docs/demo.txt`, and this file.
- **Project metadata.** `pyproject.toml`, `LICENSE`, `.gitignore`, and `.github/workflows/ci.yml`.
