"""Seed BigQuery with the sales-demo warehouse tables.

Creates five tables under ``$GCP_PROJECT_ID.$BIGQUERY_DATASET_ID``:

  - ``accounts``                — customer roster with derived ARR / health / renewal
  - ``products``                — product catalog
  - ``account_product_usage``   — monthly usage facts (MAU vs contracted seats)
  - ``account_contacts``        — named contacts per account (champion / buyer / ...)
  - ``account_renewals``        — one contract / renewal record per account

Re-runnable: every table is dropped and recreated, and the data is generated
from a fixed seed in :mod:`seed_data`, so the result is deterministic.

Run:
    uv run python import/sales_agent/seed_bigquery.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_data import build_seed_dataset  # noqa: E402

load_dotenv()


ACCOUNTS_SCHEMA = [
    bigquery.SchemaField("account_id",          "STRING", mode="REQUIRED"),
    bigquery.SchemaField("name",                "STRING", mode="REQUIRED"),
    bigquery.SchemaField("domain",              "STRING", mode="REQUIRED"),
    bigquery.SchemaField("industry",            "STRING"),
    bigquery.SchemaField("region",              "STRING"),
    bigquery.SchemaField("size_class",          "STRING"),
    bigquery.SchemaField("employee_count_band", "STRING"),
    bigquery.SchemaField("arr_band_usd",        "STRING"),
    bigquery.SchemaField("arr_usd",             "NUMERIC"),
    bigquery.SchemaField("seats_total",         "INTEGER"),
    bigquery.SchemaField("avg_utilization",     "FLOAT"),
    bigquery.SchemaField("health_score",        "INTEGER"),
    bigquery.SchemaField("trajectory",          "STRING"),
    bigquery.SchemaField("signed_at",           "DATE"),
    bigquery.SchemaField("renewal_date",        "DATE"),
    bigquery.SchemaField("owner_csm",           "STRING"),
]

PRODUCTS_SCHEMA = [
    bigquery.SchemaField("sku",            "STRING", mode="REQUIRED"),
    bigquery.SchemaField("name",           "STRING", mode="REQUIRED"),
    bigquery.SchemaField("category",       "STRING"),
    bigquery.SchemaField("tier",           "STRING"),
    bigquery.SchemaField("list_price_usd", "NUMERIC"),
    bigquery.SchemaField("launched_at",    "DATE"),
]

USAGE_SCHEMA = [
    bigquery.SchemaField("account_id",          "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("sku",                 "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("month",               "DATE",      mode="REQUIRED"),
    bigquery.SchemaField("mau",                 "INTEGER"),
    bigquery.SchemaField("monthly_revenue_usd", "NUMERIC"),
    bigquery.SchemaField("last_active_at",      "TIMESTAMP"),
    bigquery.SchemaField("contracted_seats",    "INTEGER"),
]

CONTACTS_SCHEMA = [
    bigquery.SchemaField("contact_id",        "STRING", mode="REQUIRED"),
    bigquery.SchemaField("account_id",        "STRING", mode="REQUIRED"),
    bigquery.SchemaField("first_name",        "STRING"),
    bigquery.SchemaField("last_name",         "STRING"),
    bigquery.SchemaField("email",             "STRING"),
    bigquery.SchemaField("title",             "STRING"),
    bigquery.SchemaField("role",              "STRING"),
    bigquery.SchemaField("is_decision_maker", "BOOL"),
    bigquery.SchemaField("is_champion",       "BOOL"),
]

RENEWALS_SCHEMA = [
    bigquery.SchemaField("account_id",     "STRING", mode="REQUIRED"),
    bigquery.SchemaField("contract_start", "DATE"),
    bigquery.SchemaField("term_months",    "INTEGER"),
    bigquery.SchemaField("renewal_date",   "DATE"),
    bigquery.SchemaField("arr_usd",        "NUMERIC"),
    bigquery.SchemaField("seats_total",    "INTEGER"),
    bigquery.SchemaField("auto_renew",     "BOOL"),
    bigquery.SchemaField("status",         "STRING"),
]


def _bq_client() -> bigquery.Client:
    billing = os.environ.get("GCP_BILLING_PROJECT_ID")
    project = os.environ["GCP_PROJECT_ID"]
    return bigquery.Client(project=billing or project)


def _ensure_dataset(client: bigquery.Client, dataset_id: str) -> bigquery.DatasetReference:
    project = os.environ["GCP_PROJECT_ID"]
    dataset_ref = bigquery.DatasetReference(project, dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = os.environ.get("BIGQUERY_REGION", "US")
        client.create_dataset(dataset)
        print(f"created dataset {project}.{dataset_id} in {dataset.location}")
    return dataset_ref


def _replace_table(
    client: bigquery.Client,
    dataset_ref: bigquery.DatasetReference,
    table_name: str,
    schema: list[bigquery.SchemaField],
    rows: list[dict],
) -> None:
    table_ref = dataset_ref.table(table_name)
    table = bigquery.Table(table_ref, schema=schema)
    client.delete_table(table, not_found_ok=True)
    client.create_table(table)

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    load_job = client.load_table_from_json(rows, table_ref, job_config=job_config)
    load_job.result()
    print(f"  {table_name}: loaded {len(rows)} rows")


def main() -> None:
    project = os.environ["GCP_PROJECT_ID"]
    dataset_id = os.environ["BIGQUERY_DATASET_ID"]

    print(f"seeding BigQuery: {project}.{dataset_id}")
    data = build_seed_dataset()

    client = _bq_client()
    dataset_ref = _ensure_dataset(client, dataset_id)

    _replace_table(client, dataset_ref, "accounts", ACCOUNTS_SCHEMA, data["accounts"])
    _replace_table(client, dataset_ref, "products", PRODUCTS_SCHEMA, data["products"])
    _replace_table(client, dataset_ref, "account_product_usage", USAGE_SCHEMA, data["usage"])
    _replace_table(client, dataset_ref, "account_contacts", CONTACTS_SCHEMA, data["contacts"])
    _replace_table(client, dataset_ref, "account_renewals", RENEWALS_SCHEMA, data["renewals"])

    print("done.")


if __name__ == "__main__":
    main()
