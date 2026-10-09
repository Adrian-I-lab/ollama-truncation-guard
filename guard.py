"""Pure checks that decide whether a model saw the whole prompt.

Ollama reports prompt_eval_count, the number of prompt tokens the model actually
evaluated. If that is far below what the client sent, the prompt was silently
truncated and the answer cannot be trusted. Nothing here does I/O.
"""


class TruncationError(RuntimeError):
    """Raised by callers that prefer an exception to a list of reasons."""


def estimate_tokens(text, chars_per_token=6.0):
    """Return a LOWER bound on the token count of text.

    English prose runs nearer 4 characters per token, so 6.0 is deliberately
    generous. Dividing by it under-counts, so estimation error alone cannot
    make an honest prompt look truncated.
    """
    return int(len(text) / chars_per_token)


def prompt_text(path, body):
    """Return the prompt text an Ollama request carries, for counting."""
    if path == "/api/chat":
        return "".join(str(m.get("content") or "") for m in body.get("messages") or [])
    return str(body.get("system") or "") + str(body.get("prompt") or "")


def check(prompt_eval_count, expected, num_ctx=None, tolerance=0.85):
    """Return a list of reasons to distrust the answer. An empty list means OK."""
    count = prompt_eval_count
    if count is None:
        return ["missing prompt_eval_count; truncation cannot be ruled out"]
    reasons = []
    if expected and tolerance > 0 and count < tolerance * expected:
        reasons.append(
            f"prompt_eval_count {count} is below {tolerance} x expected {expected}"
        )
    if num_ctx and abs(count - num_ctx // 2) <= 8:
        if expected is None or expected > 1.1 * count:
            reasons.append(
                f"half-window signature: prompt_eval_count {count} is about"
                f" num_ctx {num_ctx} / 2"
            )
    if num_ctx and count > num_ctx:
        reasons.append(f"count exceeds num_ctx: prompt_eval_count {count} > {num_ctx}")
    return reasons


def preflight(expected, num_ctx):
    """Return a reason if the prompt cannot fit before the model is even called."""
    if num_ctx and expected and expected > num_ctx:
        return f"expected {expected} tokens exceed num_ctx {num_ctx}"
    return None
