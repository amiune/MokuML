CREATE TABLE customers (
  id uuid PRIMARY KEY,
  created_at timestamp NOT NULL,
  country text,
  churned_next_month boolean
);

CREATE TABLE orders (
  id uuid PRIMARY KEY,
  customer_id uuid NOT NULL REFERENCES customers(id),
  total numeric NOT NULL,
  created_at timestamp NOT NULL
);

CREATE TABLE invoices (
  id uuid PRIMARY KEY,
  customer_id uuid NOT NULL REFERENCES customers(id),
  due_date date NOT NULL,
  paid_at date,
  amount numeric NOT NULL
);

