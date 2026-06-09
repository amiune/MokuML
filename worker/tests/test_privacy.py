from __future__ import annotations

import unittest

from worker.llm.prompt_builder import build_user_payload
from worker.db.connection import _safe_error
from worker.db.urls import normalize_sqlalchemy_connection_string
from worker.security.masking import mask_connection_string
from worker.security.privacy import find_llm_privacy_violations


class PrivacyTests(unittest.TestCase):
    def test_llm_payload_contains_schema_not_rows(self) -> None:
        payload = build_user_payload(
            "Predict late invoices",
            {
                "dialect": "postgres",
                "schemas": [
                    {
                        "name": "public",
                        "tables": [
                            {
                                "name": "invoices",
                                "columns": [{"name": "due_date", "type": "date"}],
                            }
                        ],
                    }
                ],
            },
        )
        self.assertEqual(find_llm_privacy_violations(payload), [])

    def test_llm_payload_rejects_preview_rows(self) -> None:
        payload = {
            "prediction_request": "Predict churn",
            "schema": {},
            "preview": [{"customer": "Acme", "revenue": 1000}],
        }
        self.assertTrue(find_llm_privacy_violations(payload))

    def test_masks_connection_password(self) -> None:
        masked = mask_connection_string("postgresql://alice:secret@localhost:5432/shop")
        self.assertIn("alice:****@", masked)
        self.assertNotIn("secret", masked)

    def test_masks_password_in_connection_errors(self) -> None:
        error = Exception("failed for postgresql://alice:secret@localhost/shop password=secret")
        safe = _safe_error(error, "postgresql://alice:secret@localhost/shop")
        self.assertNotIn("secret", safe)
        self.assertIn("****", safe)

    def test_normalizes_postgres_driver_alias(self) -> None:
        normalized = normalize_sqlalchemy_connection_string("postgresql://alice:secret@localhost/shop")
        self.assertTrue(normalized.startswith("postgresql+psycopg://"))


if __name__ == "__main__":
    unittest.main()
