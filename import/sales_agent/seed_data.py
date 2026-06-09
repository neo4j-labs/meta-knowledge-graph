"""Canonical sales-demo dataset shared by the BigQuery + Neo4j seeders.

Design goals (why this is shaped the way it is):

* **Internally consistent.** A company's ``employee_count_band``, contracted
  seats, ``arr_band_usd`` and ``arr_usd`` are all derived from one ``size_class``
  and the generated usage — they reconcile instead of being independent random
  draws. (The previous seed picked them with ``rng.choice`` independently, so an
  account could read "51-200 employees / $5m+ ARR" on top of $120k of usage.)
* **Signal-bearing.** Every account is assigned a *trajectory* archetype
  (``expanding`` / ``steady`` / ``at_risk`` / ``new``) that shapes its monthly
  usage, so the assistant can actually find expansion candidates, churn/renewal
  risk, and ramping new logos instead of every account trending up forever.
* **Actionable.** Accounts carry contacts (champion / economic buyer / ...) and a
  renewal/contract record, so "who do I email" and "what renews in 90 days" have
  answers.
* **Real where it helps.** Enterprise customer names + domains are real so
  Diffbot ``enhance_entity`` / ``search_news`` return live hits. Everything
  else is synthetic and deterministic for a fixed seed.

``build_seed_dataset()`` returns the whole dataset; the BigQuery and Neo4j
seeders consume the same dict so both stores stay in lock-step.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone
from typing import Any

SEED = 20260601

# --- people -----------------------------------------------------------------
# Weighted so Marcus Wei carries the largest book (nice for "show my pipeline").
OWNER_WEIGHTS: dict[str, float] = {
    "Marcus Wei": 0.30,
    "Priya Shah": 0.20,
    "Diego Ramirez": 0.20,
    "Sofia Müller": 0.15,
    "Hannah Park": 0.15,
}

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Jamie", "Avery",
    "Sam", "Dana", "Priya", "Wei", "Sofia", "Diego", "Hannah", "Noah", "Maya",
    "Liam", "Olivia", "Ethan", "Aisha", "Kenji", "Lucas", "Nina",
]
LAST_NAMES = [
    "Chen", "Patel", "Garcia", "Kim", "Nguyen", "Okafor", "Rossi", "Schmidt",
    "Andersson", "Silva", "Haddad", "Yamamoto", "Novak", "Ibrahim", "Costa",
    "Murphy", "Bauer", "Petrov", "Khan", "Reyes",
]

EMPLOYEE_BAND_BY_SIZE = {
    "smb": "51-200",
    "mid": "201-1000",
    "ent": "1001-5000",
    "strategic": "5001+",
}

SIZE_CLASSES = {
    "smb":       {"seat_range": (8, 45),      "n_products": (1, 3)},
    "mid":       {"seat_range": (40, 160),    "n_products": (2, 4)},
    "ent":       {"seat_range": (150, 650),   "n_products": (3, 6)},
    "strategic": {"seat_range": (500, 1600),  "n_products": (4, 8)},
}

# Curated so well-known enterprise buyers read at a believable scale next to
# Diffbot truth.
SIZE_OVERRIDES = {
    # strategic — global workforces, heavy business travel, field mobility, or both
    "accenture": "strategic", "deloitte": "strategic", "pwc": "strategic",
    "ey": "strategic", "ibm": "strategic", "microsoft": "strategic",
    "pfizer": "strategic", "novartis": "strategic", "merck": "strategic",
    "thermo-fisher": "strategic", "unitedhealth": "strategic",
    "state-farm": "strategic", "jpmorgan": "strategic",
    "bank-of-america": "strategic", "wells-fargo": "strategic",
    "walmart": "strategic", "target": "strategic", "home-depot": "strategic",
    "lowes": "strategic", "shell": "strategic", "chevron": "strategic",
    "aecom": "strategic",
    # enterprise
    "kpmg": "ent", "mckinsey": "ent", "bcg": "ent", "bain": "ent",
    "salesforce": "ent", "oracle": "ent", "sap": "ent", "cisco": "ent",
    "johnson-controls": "ent", "otis": "ent", "schindler": "ent",
    "abbott": "ent", "medtronic": "ent", "astrazeneca": "ent",
    "humana": "ent", "allstate": "ent", "liberty-mutual": "ent",
    "travelers": "ent", "best-buy": "ent", "duke-energy": "ent",
    "nextera-energy": "ent", "jacobs": "ent", "turner": "ent",
    "bechtel": "ent", "caterpillar": "ent",
    # mid-market enterprise programs
    "stanley-black-decker": "mid",
}

# Probability an account in each size carries a support / services line. Bigger
# accounts almost always have support, so a *missing* support line is a signal.
SUPPORT_PROB = {"smb": 0.30, "mid": 0.60, "ent": 0.85, "strategic": 0.95}
SERVICES_PROB = {"smb": 0.10, "mid": 0.30, "ent": 0.50, "strategic": 0.70}

# Preferred rental program by size; a minority are intentionally under-tiered to
# create realistic upgrade opportunities the assistant can surface.
PROGRAM_PREF = {
    "smb": "roadflex-business",
    "mid": "roadflex-premium",
    "ent": "roadflex-global",
    "strategic": "roadflex-global",
}
PROGRAM_DOWNGRADE = {
    "roadflex-global": "roadflex-premium",
    "roadflex-premium": "roadflex-business",
    "roadflex-business": "roadflex-business",
}
MISTIER_PROB = 0.22

TRAJECTORY_WEIGHTS = {
    "expanding": 0.30,
    "steady": 0.40,
    "at_risk": 0.20,
    "new": 0.10,
}

# (account_id, name, domain, industry, region)
ACCOUNTS_RAW: list[tuple[str, str, str, str, str]] = [
    # Professional services and consulting: high business-travel demand
    ("accenture",             "Accenture",                 "accenture.com",                "Professional Services", "Global"),
    ("deloitte",              "Deloitte",                  "deloitte.com",                 "Professional Services", "Global"),
    ("pwc",                   "PwC",                       "pwc.com",                      "Professional Services", "Global"),
    ("kpmg",                  "KPMG",                      "kpmg.com",                     "Professional Services", "Global"),
    ("ey",                    "EY",                        "ey.com",                       "Professional Services", "Global"),
    ("mckinsey",              "McKinsey & Company",        "mckinsey.com",                 "Consulting",            "Global"),
    ("bcg",                   "Boston Consulting Group",   "bcg.com",                      "Consulting",            "Global"),
    ("bain",                  "Bain & Company",            "bain.com",                     "Consulting",            "Global"),
    # Technology: sales, implementation, and customer-success travel
    ("salesforce",            "Salesforce",                "salesforce.com",               "Software",              "Global"),
    ("oracle",                "Oracle",                    "oracle.com",                   "Software",              "Global"),
    ("sap",                   "SAP",                       "sap.com",                      "Software",              "Global"),
    ("cisco",                 "Cisco",                     "cisco.com",                    "Technology",            "Global"),
    ("ibm",                   "IBM",                       "ibm.com",                      "Technology",            "Global"),
    ("microsoft",             "Microsoft",                 "microsoft.com",                "Technology",            "Global"),
    # Field service and manufacturing
    ("johnson-controls",       "Johnson Controls",          "johnsoncontrols.com",          "Field Services",        "Global"),
    ("otis",                  "Otis",                      "otis.com",                     "Field Services",        "Global"),
    ("schindler",             "Schindler",                 "schindler.com",                "Field Services",        "Global"),
    ("stanley-black-decker",   "Stanley Black & Decker",    "stanleyblackanddecker.com",    "Manufacturing",         "Global"),
    ("caterpillar",           "Caterpillar",               "caterpillar.com",              "Manufacturing",         "Global"),
    # Healthcare, pharma, and life sciences
    ("abbott",                "Abbott",                    "abbott.com",                   "Healthcare",            "Global"),
    ("medtronic",             "Medtronic",                 "medtronic.com",                "MedTech",               "Global"),
    ("thermo-fisher",         "Thermo Fisher Scientific",  "thermofisher.com",             "Life Sciences",         "Global"),
    ("pfizer",                "Pfizer",                    "pfizer.com",                   "Pharmaceuticals",       "Global"),
    ("novartis",              "Novartis",                  "novartis.com",                 "Pharmaceuticals",       "Global"),
    ("merck",                 "Merck",                     "merck.com",                    "Pharmaceuticals",       "Global"),
    ("astrazeneca",           "AstraZeneca",               "astrazeneca.com",              "Pharmaceuticals",       "Global"),
    ("unitedhealth",          "UnitedHealth Group",        "unitedhealthgroup.com",        "Healthcare",            "NA"),
    ("humana",                "Humana",                    "humana.com",                   "Healthcare",            "NA"),
    # Insurance and financial services
    ("allstate",              "Allstate",                  "allstate.com",                 "Insurance",             "NA"),
    ("state-farm",            "State Farm",                "statefarm.com",                "Insurance",             "NA"),
    ("liberty-mutual",        "Liberty Mutual",            "libertymutual.com",            "Insurance",             "NA"),
    ("travelers",             "Travelers",                 "travelers.com",                "Insurance",             "NA"),
    ("jpmorgan",              "JPMorgan Chase",            "jpmorganchase.com",            "Financial Services",    "Global"),
    ("bank-of-america",       "Bank of America",           "bankofamerica.com",            "Financial Services",    "NA"),
    ("wells-fargo",           "Wells Fargo",               "wellsfargo.com",               "Financial Services",    "NA"),
    # Retail operations and corporate travel
    ("walmart",               "Walmart",                   "walmart.com",                  "Retail",                "NA"),
    ("target",                "Target",                    "target.com",                   "Retail",                "NA"),
    ("best-buy",              "Best Buy",                  "bestbuy.com",                  "Retail",                "NA"),
    ("home-depot",            "The Home Depot",            "homedepot.com",                "Retail",                "NA"),
    ("lowes",                 "Lowe's",                    "lowes.com",                    "Retail",                "NA"),
    # Energy, utilities, engineering, and construction
    ("shell",                 "Shell",                     "shell.com",                    "Energy",                "Global"),
    ("chevron",               "Chevron",                   "chevron.com",                  "Energy",                "Global"),
    ("duke-energy",           "Duke Energy",               "duke-energy.com",              "Utilities",             "NA"),
    ("nextera-energy",        "NextEra Energy",            "nexteraenergy.com",            "Utilities",             "NA"),
    ("jacobs",                "Jacobs",                    "jacobs.com",                   "Engineering",           "Global"),
    ("aecom",                 "AECOM",                     "aecom.com",                    "Engineering",           "Global"),
    ("turner",                "Turner Construction",       "turnerconstruction.com",       "Construction",          "NA"),
    ("bechtel",               "Bechtel",                   "bechtel.com",                  "Construction",          "Global"),
]

PRODUCTS: list[dict[str, Any]] = [
    # Rental programs — recurring enterprise rental agreements
    {"sku": "roadflex-business",       "name": "RoadFlex Business Rental Program",     "category": "program",  "tier": "business",   "list_price_usd": 1800},
    {"sku": "roadflex-premium",        "name": "RoadFlex Premium Rental Program",      "category": "program",  "tier": "premium",    "list_price_usd": 3200},
    {"sku": "roadflex-global",         "name": "RoadFlex Global Mobility Program",     "category": "program",  "tier": "global",     "list_price_usd": 6200},
    {"sku": "roadflex-project-pool",   "name": "RoadFlex Project Vehicle Pool",        "category": "program",  "tier": "enterprise", "list_price_usd": 4800},
    # Add-ons that drive attach / whitespace plays
    {"sku": "damage-waiver",           "name": "Corporate Damage Waiver",              "category": "addon",    "tier": None,         "list_price_usd": 850},
    {"sku": "liability-cover",         "name": "Supplemental Liability Coverage",      "category": "addon",    "tier": None,         "list_price_usd": 950},
    {"sku": "airport-delivery",        "name": "Airport & Office Delivery",            "category": "addon",    "tier": None,         "list_price_usd": 1200},
    {"sku": "ev-hybrid-upgrade",       "name": "EV & Hybrid Vehicle Access",           "category": "addon",    "tier": None,         "list_price_usd": 1500},
    {"sku": "fuel-service",            "name": "Fuel & Charging Service",              "category": "addon",    "tier": None,         "list_price_usd": 450},
    {"sku": "mileage-package",         "name": "High-Mileage Package",                 "category": "addon",    "tier": None,         "list_price_usd": 700},
    {"sku": "corporate-billing",       "name": "Corporate Billing Integration",        "category": "addon",    "tier": None,         "list_price_usd": 900},
    {"sku": "driver-verification",     "name": "Driver Eligibility Verification",      "category": "addon",    "tier": None,         "list_price_usd": 550},
    {"sku": "roadside-plus",           "name": "Roadside Plus",                        "category": "addon",    "tier": None,         "list_price_usd": 650},
    {"sku": "one-way-rentals",         "name": "One-Way Rental Network",               "category": "addon",    "tier": None,         "list_price_usd": 800},
    {"sku": "chauffeur-network",       "name": "Executive Chauffeur Network",          "category": "addon",    "tier": None,         "list_price_usd": 2200},
    # Support
    {"sku": "support-priority",        "name": "Priority Mobility Support",            "category": "support",  "tier": "premium",    "list_price_usd": 1200},
    {"sku": "support-247",             "name": "24x7 Traveler Support",                "category": "support",  "tier": "enterprise", "list_price_usd": 4200},
    # Services
    {"sku": "implementation-pack",     "name": "Program Implementation & Onboarding",  "category": "services", "tier": None,         "list_price_usd": 3000},
    {"sku": "travel-policy-design",    "name": "Travel Policy Design",                 "category": "services", "tier": None,         "list_price_usd": 4500},
    {"sku": "mobility-analytics",      "name": "Mobility Analytics Advisory",          "category": "services", "tier": None,         "list_price_usd": 5200},
    {"sku": "beta-access",             "name": "Beta Feature Access",                  "category": "services", "tier": None,         "list_price_usd": 0},
]

LAUNCH_DATES = {
    "program":  date(2022, 1, 1),
    "addon":    date(2022, 6, 1),
    "support":  date(2022, 1, 1),
    "services": date(2023, 1, 1),
}

CONTACT_ROLES = [
    ("champion",       ["Global Travel Manager", "Mobility Program Manager", "Corporate Mobility Manager", "Travel Operations Manager"], False, True),
    ("economic_buyer", ["VP Procurement", "VP Travel & Expense", "Director of Strategic Sourcing", "VP Operations"],                  True,  False),
    ("technical",      ["Travel Systems Administrator", "Expense Platform Lead", "Accounts Payable Lead", "Mobility Data Analyst"],   False, False),
    ("executive",      ["Chief Procurement Officer", "Chief Financial Officer", "SVP Operations"],                                     True,  False),
]
N_CONTACTS_BY_SIZE = {"smb": 2, "mid": 3, "ent": 3, "strategic": 4}


def _add_months(start: date, n: int) -> date:
    total = (start.year * 12 + (start.month - 1)) + n
    return date(total // 12, total % 12 + 1, 1)


def _account_rng(account_id: str) -> random.Random:
    """Stable per-account RNG: adding/removing accounts doesn't reshuffle others."""
    return random.Random(f"{SEED}:{account_id}")


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _arr_band(arr: float) -> str:
    if arr < 100_000:
        return "<100k"
    if arr < 500_000:
        return "100k-500k"
    if arr < 1_000_000:
        return "500k-1m"
    if arr < 5_000_000:
        return "1m-5m"
    return "5m+"


def materialize_products() -> list[dict[str, Any]]:
    return [{**p, "launched_at": LAUNCH_DATES[p["category"]].isoformat()} for p in PRODUCTS]


def _products_by_category() -> dict[str, list[dict[str, Any]]]:
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for p in PRODUCTS:
        by_cat.setdefault(p["category"], []).append(p)
    return by_cat


def _select_skus(rng: random.Random, size_class: str) -> list[str]:
    by_cat = _products_by_category()
    addon_skus = [p["sku"] for p in by_cat["addon"]]
    support_skus = [p["sku"] for p in by_cat["support"]]
    services_skus = [p["sku"] for p in by_cat["services"]]

    program = PROGRAM_PREF[size_class]
    if rng.random() < MISTIER_PROB:
        program = PROGRAM_DOWNGRADE[program]

    chosen = [program]
    if rng.random() < SUPPORT_PROB[size_class]:
        chosen.append(rng.choice(support_skus))
    if rng.random() < SERVICES_PROB[size_class]:
        chosen.append(rng.choice(services_skus))

    n_total = rng.randint(*SIZE_CLASSES[size_class]["n_products"])
    remaining = max(0, n_total - len(chosen))
    chosen += rng.sample(addon_skus, min(remaining, len(addon_skus)))
    return chosen


def _mau_ratio_endpoints(rng: random.Random, trajectory: str) -> tuple[float, float]:
    if trajectory == "expanding":
        return rng.uniform(0.45, 0.65), rng.uniform(1.05, 1.45)
    if trajectory == "steady":
        start = rng.uniform(0.60, 0.82)
        return start, start + rng.uniform(-0.05, 0.08)
    if trajectory == "at_risk":
        return rng.uniform(0.75, 0.95), rng.uniform(0.28, 0.50)
    # new
    return rng.uniform(0.15, 0.30), rng.uniform(0.55, 0.80)


def materialize_accounts() -> list[dict[str, Any]]:
    """Account base records (ARR figures are filled in later from usage)."""
    accounts = []
    for account_id, name, domain, industry, region in ACCOUNTS_RAW:
        rng = _account_rng(account_id)
        size_class = SIZE_OVERRIDES.get(
            account_id,
            _weighted_choice(rng, {"smb": 0.25, "mid": 0.35, "ent": 0.30, "strategic": 0.10}),
        )
        trajectory = _weighted_choice(rng, TRAJECTORY_WEIGHTS)
        signed = date(2022, 6, 1) + timedelta(days=rng.randint(0, 1100))
        accounts.append({
            "account_id": account_id,
            "name": name,
            "domain": domain,
            "industry": industry,
            "region": region,
            "size_class": size_class,
            "employee_count_band": EMPLOYEE_BAND_BY_SIZE[size_class],
            "trajectory": trajectory,
            "signed_at": signed.isoformat(),
            "owner_csm": _weighted_choice(rng, OWNER_WEIGHTS),
            # arr_band_usd / arr_usd / health_score filled by finalize step.
        })
    return accounts


def generate_usage(
    accounts: list[dict[str, Any]],
    months_back: int = 24,
) -> list[dict[str, Any]]:
    price = {p["sku"]: p["list_price_usd"] for p in PRODUCTS}
    today = date.today().replace(day=1)
    start_month = _add_months(today, -(months_back - 1))
    rows: list[dict[str, Any]] = []

    for account in accounts:
        rng = _account_rng(account["account_id"] + ":usage")
        trajectory = account["trajectory"]
        skus = _select_skus(rng, account["size_class"])
        # "new" accounts only have a short, recent rental-program history.
        active_months = months_back if trajectory != "new" else rng.randint(6, 10)
        churn_sku = rng.choice(skus) if trajectory == "at_risk" and rng.random() < 0.5 else None

        for sku in skus:
            lo, hi = SIZE_CLASSES[account["size_class"]]["seat_range"]
            seats = rng.randint(lo, hi)
            start_ratio, end_ratio = _mau_ratio_endpoints(rng, trajectory)
            seat_mult = max(1.0, seats / 40.0)
            base_revenue = round(price[sku] * seat_mult, 2)

            for i in range(months_back):
                if i < months_back - active_months:
                    continue  # account/sku not active yet (new logos)
                month = _add_months(start_month, i)
                t = i / max(1, months_back - 1)
                ratio = (start_ratio + (end_ratio - start_ratio) * t) * rng.uniform(0.95, 1.05)
                mau = max(0, int(round(seats * ratio)))
                revenue = round(base_revenue * rng.uniform(0.85, 1.0), 2)

                # A churned line drops to zero usage + revenue in the final months.
                months_from_end = (months_back - 1) - i
                if churn_sku == sku and months_from_end <= 1:
                    mau, revenue = 0, 0.0

                last_active = datetime(
                    month.year, month.month, min(28, rng.randint(1, 28)),
                    rng.randint(8, 22), rng.randint(0, 59), tzinfo=timezone.utc,
                )
                rows.append({
                    "account_id": account["account_id"],
                    "sku": sku,
                    "month": month.isoformat(),
                    "mau": mau,
                    "monthly_revenue_usd": revenue,
                    "last_active_at": last_active.isoformat(),
                    "contracted_seats": seats,
                })
    return rows


def latest_usage_by_account(usage: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Most-recent month row per (account, sku) — the basis for graph edges."""
    latest_month: dict[tuple[str, str], str] = {}
    for r in usage:
        key = (r["account_id"], r["sku"])
        if r["month"] > latest_month.get(key, ""):
            latest_month[key] = r["month"]
    out: dict[str, list[dict[str, Any]]] = {}
    for r in usage:
        key = (r["account_id"], r["sku"])
        if r["month"] != latest_month[key]:
            continue
        seats = r["contracted_seats"] or 0
        util = round(r["mau"] / seats, 3) if seats else None
        out.setdefault(r["account_id"], []).append({**r, "utilization": util})
    return out


def _finalize_accounts(
    accounts: list[dict[str, Any]],
    usage: list[dict[str, Any]],
) -> None:
    """Derive ARR / health / contracted vehicle totals from real usage."""
    latest = latest_usage_by_account(usage)
    for account in accounts:
        lines = latest.get(account["account_id"], [])
        mrr = sum(line["monthly_revenue_usd"] for line in lines)
        seats_total = sum(line["contracted_seats"] for line in lines)
        utils = [line["utilization"] for line in lines if line["utilization"] is not None]
        avg_util = sum(utils) / len(utils) if utils else 0.0
        arr = round(mrr * 12, 2)

        base = {"expanding": 84, "steady": 73, "new": 64, "at_risk": 38}[account["trajectory"]]
        rng = _account_rng(account["account_id"] + ":health")
        health = int(max(5, min(99, base + (avg_util - 0.8) * 15 + rng.randint(-5, 5))))

        account["arr_usd"] = arr
        account["arr_band_usd"] = _arr_band(arr)
        account["seats_total"] = seats_total
        account["avg_utilization"] = round(avg_util, 3)
        account["health_score"] = health


def generate_contacts(accounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for account in accounts:
        rng = _account_rng(account["account_id"] + ":contacts")
        n = N_CONTACTS_BY_SIZE[account["size_class"]]
        for idx in range(n):
            role, titles, is_dm, is_champ = CONTACT_ROLES[idx % len(CONTACT_ROLES)]
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            contacts.append({
                "contact_id": f"{account['account_id']}-c{idx + 1}",
                "account_id": account["account_id"],
                "first_name": first,
                "last_name": last,
                "email": f"{first}.{last}@{account['domain']}".lower(),
                "title": rng.choice(titles),
                "role": role,
                "is_decision_maker": is_dm,
                "is_champion": is_champ,
            })
    return contacts


def generate_renewals(accounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    today = date.today()
    renewals: list[dict[str, Any]] = []
    for account in accounts:
        rng = _account_rng(account["account_id"] + ":renewal")
        # at_risk accounts cluster their renewal in the near term (the squeeze).
        if account["trajectory"] == "at_risk":
            offset = rng.randint(5, 75)
            auto_renew = rng.random() < 0.35
        elif account["trajectory"] == "new":
            offset = rng.randint(180, 420)
            auto_renew = rng.random() < 0.8
        else:
            offset = rng.randint(20, 360)
            auto_renew = rng.random() < 0.7
        renewal_date = today + timedelta(days=offset)
        term_months = rng.choice([12, 12, 12, 24, 36])
        renewals.append({
            "account_id": account["account_id"],
            "contract_start": account["signed_at"],
            "term_months": term_months,
            "renewal_date": renewal_date.isoformat(),
            "arr_usd": account["arr_usd"],
            "seats_total": account["seats_total"],
            "auto_renew": auto_renew,
            "status": "active",
        })
    return renewals


def build_seed_dataset(months_back: int = 24) -> dict[str, list[dict[str, Any]]]:
    accounts = materialize_accounts()
    products = materialize_products()
    usage = generate_usage(accounts, months_back=months_back)
    _finalize_accounts(accounts, usage)
    contacts = generate_contacts(accounts)
    renewals = generate_renewals(accounts)
    # Denormalize renewal_date onto accounts so single-table account queries
    # (SQL or Cypher) can answer "what renews soon" without a join.
    renewal_by_acct = {r["account_id"]: r for r in renewals}
    for account in accounts:
        account["renewal_date"] = renewal_by_acct[account["account_id"]]["renewal_date"]
    return {
        "accounts": accounts,
        "products": products,
        "usage": usage,
        "contacts": contacts,
        "renewals": renewals,
    }


if __name__ == "__main__":
    from collections import Counter

    data = build_seed_dataset()
    print(f"accounts : {len(data['accounts'])}")
    print(f"products : {len(data['products'])}")
    print(f"usage    : {len(data['usage'])} rows")
    print(f"contacts : {len(data['contacts'])}")
    print(f"renewals : {len(data['renewals'])}")
    print("trajectory:", dict(Counter(a["trajectory"] for a in data["accounts"])))
    print("size_class:", dict(Counter(a["size_class"] for a in data["accounts"])))
    print("arr_band  :", dict(Counter(a["arr_band_usd"] for a in data["accounts"])))
    soon = [r for r in data["renewals"]
            if (date.fromisoformat(r["renewal_date"]) - date.today()).days <= 90]
    print(f"renewals <=90d: {len(soon)}")
    sample = next(a for a in data["accounts"] if a["account_id"] == "accenture")
    print("sample (accenture):", {k: sample[k] for k in
          ("size_class", "employee_count_band", "trajectory",
           "arr_band_usd", "arr_usd", "health_score", "owner_csm")})
