from __future__ import annotations

from typing import Any, Dict, Iterable, List


FORBIDDEN_LLM_KEYS = {
    "connection_string",
    "password",
    "api_key",
    "preview",
    "preview_rows",
    "rows",
    "records",
    "dataframe",
    "query_results",
    "cell_values",
}


def find_llm_privacy_violations(payload: Dict[str, Any]) -> List[str]:
    """Return privacy violations for an LLM payload.

    The LLM contract allows schema metadata and the natural-language request only.
    This check intentionally rejects common containers that could hold raw data.
    """
    violations: List[str] = []
    for path, key, value in _walk(payload):
        lower_key = key.lower()
        if lower_key in FORBIDDEN_LLM_KEYS:
            violations.append(f"forbidden key `{path}`")
        if lower_key.endswith("_sample") or lower_key.endswith("_samples"):
            violations.append(f"sample-like key `{path}`")
        if lower_key in {"min", "max", "distinct", "example", "examples"}:
            violations.append(f"value-profile key `{path}`")
        if isinstance(value, str) and "://" in value and "schema" not in lower_key:
            if "postgres" in value or "mysql" in value:
                violations.append(f"connection-like string at `{path}`")
    return violations


def assert_llm_payload_is_private(payload: Dict[str, Any]) -> None:
    violations = find_llm_privacy_violations(payload)
    if violations:
        joined = "; ".join(violations)
        raise ValueError(f"LLM payload violates privacy boundary: {joined}")


def _walk(value: Any, prefix: str = "") -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, str(key), child
            yield from _walk(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{prefix}[{index}]")

