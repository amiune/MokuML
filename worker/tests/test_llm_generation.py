from __future__ import annotations

import unittest
from unittest.mock import patch

from worker.llm.output_parser import parse_llm_sql_response
from worker.llm.providers import generate_sql_with_provider


SCHEMA = {
    "dialect": "postgres",
    "schemas": [
        {
            "name": "public",
            "tables": [
                {
                    "name": "customers",
                    "columns": [
                        {"name": "id", "type": "uuid"},
                        {"name": "churned", "type": "boolean"},
                    ],
                }
            ],
        }
    ],
}


class LLMGenerationTests(unittest.TestCase):
    def test_local_ollama_falls_back_to_offline_template(self) -> None:
        with patch("worker.llm.providers._call_ollama", side_effect=RuntimeError("connection refused")):
            result = generate_sql_with_provider(
                {"prediction_request": "Predict churn", "schema": SCHEMA},
                {"type": "local_ollama"},
            )

        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("SELECT * FROM public.customers LIMIT 1000", result["sql"])
        self.assertTrue(any("Local Ollama" in item for item in result["assumptions"]))

    def test_parser_accepts_json_code_fence(self) -> None:
        result = parse_llm_sql_response(
            """```json
            {
              "status": "ok",
              "task_type": "classification",
              "target_column": "churned",
              "entity_id_columns": ["id"],
              "sql": "SELECT id, churned FROM customers",
              "reasoning_summary": "Uses customer labels.",
              "leakage_warnings": [],
              "assumptions": []
            }
            ```"""
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["target_column"], "churned")


if __name__ == "__main__":
    unittest.main()
