# Synthetic examples

Every record here is fictitious and intended for software testing.

- synthetic_house_prices.csv: labeled prices with two blank outcomes for scoring. Select price_eur as regression outcome and exclude house_id from features.
- synthetic_patient_cases.csv: repeated synthetic patients with arbitrary class_a/class_b labels. Select classification, keep case_id as the prediction unit, and use patient_id for grouped evaluation. These labels have no clinical meaning.
- synthetic_churn.csv: an additional small classification example.
- postgres_demo_schema.sql: a relational demonstration schema for an explicitly disposable database.

Future-window recipes and hand-checked joins/counts/calendar boundaries are demonstrated in the test fixtures. No example establishes real-world accuracy.
