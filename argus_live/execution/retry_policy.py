from __future__ import annotations


def is_retryable_error(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ("timeout", "temporar", "rate limit", "connection", "network"))
