from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .output_parser import parse_llm_sql_response
from .prompt_builder import build_chat_messages


def generate_sql_with_provider(
    user_payload: Dict[str, Any],
    provider_config: Dict[str, Any],
) -> Dict[str, Any]:
    provider_type = provider_config.get("type", "local_ollama")
    messages = build_chat_messages(user_payload)

    if provider_type == "offline_template":
        return _offline_template_response(user_payload)
    if provider_type == "local_ollama":
        try:
            raw = _call_ollama(messages, provider_config)
            return parse_llm_sql_response(raw)
        except Exception as exc:
            return _offline_template_response(
                user_payload,
                fallback_reason=f"Local Ollama was unavailable or returned invalid output: {_short_error(exc)}",
            )
    if provider_type == "openai_compatible":
        raw = _call_openai_compatible(messages, provider_config)
        return parse_llm_sql_response(raw)

    raise ValueError(f"Unsupported LLM provider type: {provider_type}")


def _offline_template_response(user_payload: Dict[str, Any], fallback_reason: Optional[str] = None) -> Dict[str, Any]:
    schema = user_payload.get("schema", {})
    first_table = _first_table_name(schema)
    if not first_table:
        assumptions = ["Configure a local or API LLM to generate production SQL."]
        if fallback_reason:
            assumptions.append(fallback_reason)
        return {
            "status": "needs_clarification",
            "task_type": "unknown",
            "target_column": None,
            "entity_id_columns": [],
            "sql": None,
            "reasoning_summary": "No tables were available in the schema metadata.",
            "leakage_warnings": [],
            "assumptions": assumptions,
        }

    assumptions = ["No schema rows or data values were used."]
    if fallback_reason:
        assumptions.append(fallback_reason)
    return {
        "status": "needs_clarification",
        "task_type": "unknown",
        "target_column": None,
        "entity_id_columns": [],
        "sql": f"SELECT * FROM {first_table} LIMIT 1000",
        "reasoning_summary": "Offline template mode can only draft a preview query; it cannot infer a prediction target.",
        "leakage_warnings": ["Review the query and choose a target column before running ML."],
        "assumptions": assumptions,
    }


def _first_table_name(schema: Dict[str, Any]) -> str:
    for schema_entry in schema.get("schemas", []):
        schema_name = schema_entry.get("name", "public")
        for table in schema_entry.get("tables", []):
            table_name = table.get("name")
            if table_name:
                return f"{schema_name}.{table_name}" if schema_name else table_name
    return ""


def _short_error(exc: Exception, max_length: int = 240) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    text = " ".join(text.split())
    if len(text) > max_length:
        return text[: max_length - 1] + "..."
    return text


def _call_ollama(messages: List[Dict[str, str]], provider_config: Dict[str, Any]) -> str:
    endpoint = provider_config.get("endpoint", "http://127.0.0.1:11434").rstrip("/")
    model = provider_config.get("model", "sqlcoder")
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "top_p": 1.0},
    }
    response = _post_json(f"{endpoint}/api/chat", body)
    content = response.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Ollama response did not include message.content")
    return content


def _call_openai_compatible(messages: List[Dict[str, str]], provider_config: Dict[str, Any]) -> str:
    endpoint = provider_config.get("base_url", "https://api.openai.com/v1").rstrip("/")
    model = provider_config.get("model")
    api_key = provider_config.get("api_key")
    if not model:
        raise ValueError("OpenAI-compatible provider requires model")
    if not api_key:
        raise ValueError("OpenAI-compatible provider requires api_key")

    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": {"type": "json_object"},
    }
    response = _post_json(
        f"{endpoint}/chat/completions",
        body,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("OpenAI-compatible response had no choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("OpenAI-compatible response did not include message.content")
    return content


def _post_json(url: str, body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = Request(url, data=data, headers=request_headers, method="POST")
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM provider returned HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"LLM provider is unavailable: {exc.reason}") from exc
