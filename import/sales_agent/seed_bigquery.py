"""Seed BigQuery with the RoadFlex enterprise car-rental demo warehouse tables.

Creates five tables under ``$GCP_PROJECT_ID.$BIGQUERY_DATASET_ID``:

  - ``accounts``                — customer roster with derived ARR / health / renewal
  - ``products``                — product catalog
  - ``account_product_usage``   — monthly usage facts (MAU vs contracted seats)
  - ``account_contacts``        — named contacts per account (champion / buyer / ...)
  - ``account_renewals``        — one contract / renewal record per account

The dataset, every table, and every column are created with descriptions, so the
schema is self-documenting in the BigQuery console and through metadata /
neocarta context search.

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


DATASET_DESCRIPTION = (
    "RoadFlex enterprise car-rental sales / customer-success demo warehouse — "
    "system-of-record for ~48 corporate mobility accounts. Tables: accounts, products, "
    "account_product_usage (monthly time series), account_contacts, "
    "account_renewals. Mirrors the Neo4j relationship graph; join on account_id "
    "(slug) or domain."
)


ACCOUNTS_DESCRIPTION = (
    "One row per RoadFlex enterprise car-rental customer account. System-of-record "
    "roster: firmographics, derived ARR / health / rental utilization, trajectory, "
    "renewal date, and owning CSM. Join on account_id (slug) or domain; mirrors "
    "Neo4j (:Account)."
)
ACCOUNTS_SCHEMA = [
    bigquery.SchemaField(
        "account_id", "STRING", mode="REQUIRED",
        description="Account slug and primary key (e.g. 'accenture'); join key to "
                    "every other table and to Neo4j (:Account).",
    ),
    bigquery.SchemaField(
        "name", "STRING", mode="REQUIRED",
        description="Company display name (e.g. 'Accenture').",
    ),
    bigquery.SchemaField(
        "domain", "STRING", mode="REQUIRED",
        description="Primary web domain (e.g. 'accenture.com'); secondary join key "
                    "and the key Diffbot enrichment resolves on.",
    ),
    bigquery.SchemaField(
        "industry", "STRING",
        description="Vertical segment (e.g. 'Professional Services', "
                    "'Pharmaceuticals', 'Insurance', 'Retail', 'Utilities').",
    ),
    bigquery.SchemaField(
        "region", "STRING",
        description="Geographic region (e.g. 'NA' for North America).",
    ),
    bigquery.SchemaField(
        "size_class", "STRING",
        description="Account-size archetype: 'smb' / 'mid' / 'ent' / "
                    "'strategic'; drives seat range, product count, and ARR scale.",
    ),
    bigquery.SchemaField(
        "employee_count_band", "STRING",
        description="Headcount bucket derived from size_class ('51-200', "
                    "'201-1000', '1001-5000', '5001+').",
    ),
    bigquery.SchemaField(
        "arr_band_usd", "STRING",
        description="ARR bucket derived from arr_usd: '<100k', '100k-500k', "
                    "'500k-1m', '1m-5m', '5m+'.",
    ),
    bigquery.SchemaField(
        "arr_usd", "NUMERIC",
        description="Annual recurring revenue (USD): latest-month revenue across "
                    "all product lines x 12.",
    ),
    bigquery.SchemaField(
        "seats_total", "INTEGER",
        description="Total contracted rental-vehicle capacity across product "
                    "lines; one seat is one reserved vehicle.",
    ),
    bigquery.SchemaField(
        "avg_utilization", "FLOAT",
        description="Mean rental activation across lines = monthly active rental "
                    "vehicles (mau) / contracted vehicle capacity in the latest "
                    "month; >=1.0 signals a true-up.",
    ),
    bigquery.SchemaField(
        "health_score", "INTEGER",
        description="Account health 0-100 derived from trajectory + utilization; "
                    "<50 flags renewal risk.",
    ),
    bigquery.SchemaField(
        "trajectory", "STRING",
        description="Momentum archetype shaping the usage series: 'expanding', "
                    "'steady', 'at_risk', or 'new'.",
    ),
    bigquery.SchemaField(
        "signed_at", "DATE",
        description="Date the account first signed (initial contract start).",
    ),
    bigquery.SchemaField(
        "renewal_date", "DATE",
        description="Next contract renewal date (denormalized from "
                    "account_renewals for single-table queries).",
    ),
    bigquery.SchemaField(
        "owner_csm", "STRING",
        description="Name of the Customer Success Manager who owns the account "
                    "(Neo4j (:CSM)-[:OWNS]->).",
    ),
]

PRODUCTS_DESCRIPTION = (
    "RoadFlex product catalog: enterprise rental programs, attachable mobility "
    "add-ons, support plans, and services. One row per sku."
)
PRODUCTS_SCHEMA = [
    bigquery.SchemaField(
        "sku", "STRING", mode="REQUIRED",
        description="Product identifier and primary key (e.g. 'roadflex-global'); "
                    "join key to account_product_usage.sku.",
    ),
    bigquery.SchemaField(
        "name", "STRING", mode="REQUIRED",
        description="Human-readable product name.",
    ),
    bigquery.SchemaField(
        "category", "STRING",
        description="Product family: 'program' (enterprise rental agreements), "
                    "'addon' (attachable mobility options), 'support', or 'services'.",
    ),
    bigquery.SchemaField(
        "tier", "STRING",
        description="Tier within the category where applicable "
                    "('business'/'premium'/'global'/'enterprise'); NULL for add-ons "
                    "and services.",
    ),
    bigquery.SchemaField(
        "list_price_usd", "NUMERIC",
        description="Monthly list price (USD), pre-discount. Rental programs and "
                    "add-ons scale with contracted vehicle capacity; support and "
                    "services are account-level lines.",
    ),
    bigquery.SchemaField(
        "launched_at", "DATE",
        description="Date the product became generally available.",
    ),
]

USAGE_DESCRIPTION = (
    "Monthly usage facts, one row per (account, product, month). The time series "
    "behind trends, utilization, and churn detection: active rental vehicles "
    "(mau) vs contracted rental-vehicle capacity and recognized revenue."
)
USAGE_SCHEMA = [
    bigquery.SchemaField(
        "account_id", "STRING", mode="REQUIRED",
        description="Account slug; foreign key to accounts.account_id.",
    ),
    bigquery.SchemaField(
        "sku", "STRING", mode="REQUIRED",
        description="Product identifier; foreign key to products.sku.",
    ),
    bigquery.SchemaField(
        "month", "DATE", mode="REQUIRED",
        description="First day of the usage month (monthly grain) — the time "
                    "axis for trend and cohort queries.",
    ),
    bigquery.SchemaField(
        "mau", "INTEGER",
        description="Monthly active rental vehicles for this account-product-month "
                    "(the active count behind utilization).",
    ),
    bigquery.SchemaField(
        "monthly_revenue_usd", "NUMERIC",
        description="Recognized revenue (USD) for this product line in this month.",
    ),
    bigquery.SchemaField(
        "last_active_at", "TIMESTAMP",
        description="Timestamp of the last recorded activity within the month.",
    ),
    bigquery.SchemaField(
        "contracted_seats", "INTEGER",
        description="Contracted seats = reserved rental-vehicle capacity for this "
                    "line; the denominator of utilization.",
    ),
]

CONTACTS_DESCRIPTION = (
    "Named contacts per account with buying role (champion / economic buyer / "
    "technical / executive) and decision-maker / champion flags. Powers 'who do "
    "I email'."
)
CONTACTS_SCHEMA = [
    bigquery.SchemaField(
        "contact_id", "STRING", mode="REQUIRED",
        description="Contact primary key (e.g. 'accenture-c1').",
    ),
    bigquery.SchemaField(
        "account_id", "STRING", mode="REQUIRED",
        description="Owning account; foreign key to accounts.account_id.",
    ),
    bigquery.SchemaField(
        "first_name", "STRING",
        description="Contact first name.",
    ),
    bigquery.SchemaField(
        "last_name", "STRING",
        description="Contact last name.",
    ),
    bigquery.SchemaField(
        "email", "STRING",
        description="Contact email (synthetic, on the account domain).",
    ),
    bigquery.SchemaField(
        "title", "STRING",
        description="Job title (e.g. 'Global Travel Manager', 'VP Procurement').",
    ),
    bigquery.SchemaField(
        "role", "STRING",
        description="Buying-role archetype: 'champion', 'economic_buyer', "
                    "'technical', or 'executive'.",
    ),
    bigquery.SchemaField(
        "is_decision_maker", "BOOL",
        description="TRUE if the contact holds budget/decision authority "
                    "(economic buyer or executive).",
    ),
    bigquery.SchemaField(
        "is_champion", "BOOL",
        description="TRUE if the contact is the internal champion for RoadFlex.",
    ),
]

RENEWALS_DESCRIPTION = (
    "One contract / renewal record per account: term, renewal date, ARR, seats, "
    "and auto-renew flag. Powers 'what renews in N days' and renewal-risk plays."
)
RENEWALS_SCHEMA = [
    bigquery.SchemaField(
        "account_id", "STRING", mode="REQUIRED",
        description="Account slug; foreign key to accounts.account_id (one "
                    "renewal record per account).",
    ),
    bigquery.SchemaField(
        "contract_start", "DATE",
        description="Start date of the current contract term (mirrors "
                    "accounts.signed_at).",
    ),
    bigquery.SchemaField(
        "term_months", "INTEGER",
        description="Contract term length in months (12 / 24 / 36).",
    ),
    bigquery.SchemaField(
        "renewal_date", "DATE",
        description="Date the current term ends/renews; <=90 days plus risk "
                    "signals = renewal play.",
    ),
    bigquery.SchemaField(
        "arr_usd", "NUMERIC",
        description="Annual recurring revenue on the contract (USD).",
    ),
    bigquery.SchemaField(
        "seats_total", "INTEGER",
        description="Total contracted rental-vehicle capacity on the contract.",
    ),
    bigquery.SchemaField(
        "auto_renew", "BOOL",
        description="TRUE if the contract auto-renews absent action.",
    ),
    bigquery.SchemaField(
        "status", "STRING",
        description="Contract status ('active' in the seed).",
    ),
]


def _bq_client() -> bigquery.Client:
    billing = os.environ.get("GCP_BILLING_PROJECT_ID")
    project = os.environ["GCP_PROJECT_ID"]
    return bigquery.Client(project=billing or project)


def _ensure_dataset(client: bigquery.Client, dataset_id: str) -> bigquery.DatasetReference:
    project = os.environ["GCP_PROJECT_ID"]
    dataset_ref = bigquery.DatasetReference(project, dataset_id)
    try:
        dataset = client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = os.environ.get("BIGQUERY_REGION", "US")
        dataset.description = DATASET_DESCRIPTION
        client.create_dataset(dataset)
        print(f"created dataset {project}.{dataset_id} in {dataset.location}")
        return dataset_ref
    # Existing dataset: keep its description in sync with the seed.
    if dataset.description != DATASET_DESCRIPTION:
        dataset.description = DATASET_DESCRIPTION
        client.update_dataset(dataset, ["description"])
    return dataset_ref


def _replace_table(
    client: bigquery.Client,
    dataset_ref: bigquery.DatasetReference,
    table_name: str,
    schema: list[bigquery.SchemaField],
    rows: list[dict],
    description: str | None = None,
) -> None:
    table_ref = dataset_ref.table(table_name)
    table = bigquery.Table(table_ref, schema=schema)
    if description:
        table.description = description
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

    _replace_table(client, dataset_ref, "accounts", ACCOUNTS_SCHEMA, data["accounts"], ACCOUNTS_DESCRIPTION)
    _replace_table(client, dataset_ref, "products", PRODUCTS_SCHEMA, data["products"], PRODUCTS_DESCRIPTION)
    _replace_table(client, dataset_ref, "account_product_usage", USAGE_SCHEMA, data["usage"], USAGE_DESCRIPTION)
    _replace_table(client, dataset_ref, "account_contacts", CONTACTS_SCHEMA, data["contacts"], CONTACTS_DESCRIPTION)
    _replace_table(client, dataset_ref, "account_renewals", RENEWALS_SCHEMA, data["renewals"], RENEWALS_DESCRIPTION)

    print("done.")


if __name__ == "__main__":
    main()
