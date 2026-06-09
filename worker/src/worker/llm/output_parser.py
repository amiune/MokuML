from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


REQUIRED_FIELDS = {
    "status",
    "task_type",
    "target_column",
    "entity_id_columns",
    "sql",
    "reasoning_summary",
    "leakage_warnings",
    "assumptions",
}


def parse_llm_sql_response(raw_content: str) -> Dict[str, Any]:
    content = _extract_json_object(raw_content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON output: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")

    missing = REQUIRED_FIELDS.difference(parsed)
    if missing:
        raise ValueError(f"LLM response missing required fields: {sorted(missing)}")

    status = parsed.get("status")
    if status not in {"ok", "needs_clarification", "impossible"}:
        raise ValueError("LLM response status must be ok, needs_clarification, or impossible")

    task_type = parsed.get("task_type")
    if task_type not in {"classification", "regression", "unknown"}:
        raise ValueError("LLM response task_type must be classification, regression, or unknown")

    if parsed.get("target_column") is not None and not isinstance(parsed.get("target_column"), str):
        raise ValueError("LLM response target_column must be a string or null")

    _require_string_list(parsed, "entity_id_columns")
    _require_string_list(parsed, "leakage_warnings")
    _require_string_list(parsed, "assumptions")

    if parsed.get("sql") is not None and not isinstance(parsed.get("sql"), str):
        raise ValueError("LLM response sql must be a string or null")

    return parsed


def _extract_json_object(raw_content: str) -> str:
    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        return content[start : end + 1]
    return content


def _require_string_list(parsed: Dict[str, Any], key: str) -> None:
    value: Optional[List[Any]] = parsed.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"LLM response {key} must be a list of strings")
