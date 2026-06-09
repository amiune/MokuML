from __future__ import annotations

import unittest

from worker.sql.validator import validate_sql


SCHEMA = {
    "dialect": "postgres",
    "schemas": [
        {
            "name": "public",
            "tables": [
                {"name": "customers", "columns": [{"name": "id", "type": "uuid"}]},
                {"name": "orders", "columns": [{"name": "customer_id", "type": "uuid"}]},
                {"name": "invoices", "columns": [{"name": "customer_id", "type": "uuid"}]},
            ],
        }
    ],
}


class SQLValidatorTests(unittest.TestCase):
    def test_rejects_dangerous_sql(self) -> None:
        dangerous = [
            "DELETE FROM customers;",
            "INSERT INTO x VALUES (1);",
            "UPDATE customers SET name = 'x';",
            "DROP TABLE customers;",
            "ALTER TABLE customers ADD COLUMN x int;",
            "CREATE TABLE temp_x AS SELECT * FROM customers;",
            "TRUNCATE customers;",
            "SELECT * FROM customers; DELETE FROM customers;",
            "COPY customers TO '/tmp/customers.csv';",
            "CALL dangerous_proc();",
            "DO $$ BEGIN DELETE FROM customers; END $$;",
            "SELECT pg_read_file('/etc/passwd');",
        ]
        for sql in dangerous:
            with self.subTest(sql=sql):
                result = validate_sql(sql, schema=SCHEMA)
                self.assertFalse(result.is_safe)
                self.assertTrue(result.errors)

    def test_allows_safe_select(self) -> None:
        sql = """
        SELECT customer_id, COUNT(*) AS order_count
        FROM orders
        GROUP BY customer_id
        """
        result = validate_sql(sql, schema=SCHEMA)
        self.assertTrue(result.is_safe, result.errors)
        self.assertIn("LIMIT", result.normalized_sql or "")

    def test_allows_safe_with_query(self) -> None:
        sql = """
        WITH invoice_features AS (
          SELECT customer_id, COUNT(*) AS invoice_count
          FROM invoices
          GROUP BY customer_id
        )
        SELECT * FROM invoice_features
        """
        result = validate_sql(sql, schema=SCHEMA)
        self.assertTrue(result.is_safe, result.errors)

    def test_rejects_unknown_table(self) -> None:
        result = validate_sql("SELECT * FROM payroll", schema=SCHEMA)
        self.assertFalse(result.is_safe)


if __name__ == "__main__":
    unittest.main()

