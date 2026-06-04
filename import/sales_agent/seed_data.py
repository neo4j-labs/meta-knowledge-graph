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
* **Real where it helps.** Company names + domains are real so Diffbot
  ``enhance_entity`` / ``search_news`` return live hits. Everything else is
  synthetic and deterministic for a fixed seed.

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
    "smb":       {"seat_range": (5, 40),    "n_products": (1, 3)},
    "mid":       {"seat_range": (25, 120),  "n_products": (2, 4)},
    "ent":       {"seat_range": (80, 350),  "n_products": (3, 6)},
    "strategic": {"seat_range": (200, 800), "n_products": (4, 8)},
}

# Curated so well-known names read at a believable scale next to Diffbot truth.
SIZE_OVERRIDES = {
    # strategic — national fleets, thousands of power units
    "sysco": "strategic", "us-foods": "strategic", "grainger": "strategic",
    "old-dominion": "strategic", "jb-hunt": "strategic", "ryder": "strategic",
    "united-rentals": "strategic", "republic-services": "strategic",
    "cintas": "strategic",
    # enterprise
    "pfg": "ent", "gordon-food": "ent", "mclane": "ent",
    "coke-consolidated": "ent", "reyes": "ent", "keurig-dr-pepper": "ent",
    "fastenal": "ent", "ferguson": "ent", "wesco": "ent", "xpo": "ent",
    "schneider": "ent", "werner": "ent", "knight-swift": "ent",
    "sunbelt": "ent", "vulcan": "ent", "kroger": "ent", "aramark": "ent",
    "sherwin-williams": "ent",
    # mid-market
    "watsco": "mid", "saia": "mid", "estes": "mid", "arcbest": "mid",
    "penske": "mid", "martin-marietta": "mid", "cemex-usa": "mid",
    "builders-firstsource": "mid", "waste-connections": "mid",
    "clean-harbors": "mid", "stericycle": "mid", "unifirst": "mid",
    "abm": "mid", "rollins": "mid", "terminix": "mid", "brightview": "mid",
    # smaller / regional
    "casella": "smb", "trugreen": "smb", "abc-supply": "smb",
    "herc": "smb", "comfort-systems": "smb",
}

# Probability an account in each size carries a support / services line. Bigger
# accounts almost always have support, so a *missing* support line is a signal.
SUPPORT_PROB = {"smb": 0.30, "mid": 0.60, "ent": 0.85, "strategic": 0.95}
SERVICES_PROB = {"smb": 0.10, "mid": 0.30, "ent": 0.50, "strategic": 0.70}

# Preferred platform tier by size; a minority are intentionally under-tiered to
# create realistic upgrade opportunities the assistant can surface.
PLATFORM_PREF = {
    "smb": "fleetwise-starter",
    "mid": "fleetwise-pro",
    "ent": "fleetwise-enterprise",
    "strategic": "fleetwise-enterprise",
}
PLATFORM_DOWNGRADE = {
    "fleetwise-enterprise": "fleetwise-pro",
    "fleetwise-pro": "fleetwise-starter",
    "fleetwise-starter": "fleetwise-starter",
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
    # Food distribution
    ("sysco",                "Sysco",                     "sysco.com",             "Food Distribution",       "NA"),
    ("us-foods",             "US Foods",                  "usfoods.com",           "Food Distribution",       "NA"),
    ("pfg",                  "Performance Food Group",    "pfgc.com",              "Food Distribution",       "NA"),
    ("gordon-food",          "Gordon Food Service",       "gfs.com",               "Food Distribution",       "NA"),
    ("mclane",               "McLane Company",            "mclaneco.com",          "Food Distribution",       "NA"),
    # Beverage
    ("coke-consolidated",    "Coca-Cola Consolidated",    "cokeconsolidated.com",  "Beverage",                "NA"),
    ("reyes",                "Reyes Holdings",            "reyesholdings.com",     "Beverage",                "NA"),
    ("keurig-dr-pepper",     "Keurig Dr Pepper",          "keurigdrpepper.com",    "Beverage",                "NA"),
    # Industrial distribution
    ("grainger",             "W.W. Grainger",             "grainger.com",          "Industrial Distribution", "NA"),
    ("fastenal",             "Fastenal",                  "fastenal.com",          "Industrial Distribution", "NA"),
    ("ferguson",             "Ferguson",                  "ferguson.com",          "Industrial Distribution", "NA"),
    ("wesco",                "WESCO International",        "wesco.com",             "Industrial Distribution", "NA"),
    ("watsco",               "Watsco",                    "watsco.com",            "Industrial Distribution", "NA"),
    # LTL freight & trucking
    ("old-dominion",         "Old Dominion Freight Line", "odfl.com",              "LTL Freight",             "NA"),
    ("xpo",                  "XPO",                       "xpo.com",               "LTL Freight",             "NA"),
    ("saia",                 "Saia",                      "saia.com",              "LTL Freight",             "NA"),
    ("estes",                "Estes Express Lines",       "estes-express.com",     "LTL Freight",             "NA"),
    ("arcbest",              "ArcBest",                   "arcb.com",              "LTL Freight",             "NA"),
    ("schneider",            "Schneider National",        "schneider.com",         "Trucking",                "NA"),
    ("werner",               "Werner Enterprises",        "werner.com",            "Trucking",                "NA"),
    ("knight-swift",         "Knight-Swift",              "knight-swift.com",      "Trucking",                "NA"),
    # 3PL & fleet leasing
    ("jb-hunt",              "J.B. Hunt",                 "jbhunt.com",            "Logistics",               "NA"),
    ("ryder",                "Ryder System",              "ryder.com",             "Logistics",               "NA"),
    ("penske",               "Penske Truck Leasing",      "gopenske.com",          "Logistics",               "NA"),
    # Equipment rental
    ("united-rentals",       "United Rentals",            "unitedrentals.com",     "Equipment Rental",        "NA"),
    ("sunbelt",              "Sunbelt Rentals",           "sunbeltrentals.com",    "Equipment Rental",        "NA"),
    ("herc",                 "Herc Rentals",              "hercrentals.com",       "Equipment Rental",        "NA"),
    # Construction materials & building products
    ("vulcan",               "Vulcan Materials",          "vulcanmaterials.com",   "Construction Materials",  "NA"),
    ("martin-marietta",      "Martin Marietta",           "martinmarietta.com",    "Construction Materials",  "NA"),
    ("cemex-usa",            "Cemex USA",                 "cemexusa.com",          "Construction Materials",  "NA"),
    ("builders-firstsource", "Builders FirstSource",      "bldr.com",              "Building Products",       "NA"),
    ("abc-supply",           "ABC Supply",                "abcsupply.com",         "Building Products",       "NA"),
    # Waste & environmental services
    ("republic-services",    "Republic Services",         "republicservices.com",  "Waste Services",          "NA"),
    ("waste-connections",    "Waste Connections",         "wasteconnections.com",  "Waste Services",          "NA"),
    ("casella",              "Casella Waste Systems",     "casella.com",           "Waste Services",          "NA"),
    ("clean-harbors",        "Clean Harbors",             "cleanharbors.com",      "Environmental Services",  "NA"),
    ("stericycle",           "Stericycle",                "stericycle.com",        "Environmental Services",  "NA"),
    # Route-based facility & field services
    ("cintas",               "Cintas",                    "cintas.com",            "Facility Services",       "NA"),
    ("aramark",              "Aramark",                   "aramark.com",           "Facility Services",       "NA"),
    ("unifirst",             "UniFirst",                  "unifirst.com",          "Facility Services",       "NA"),
    ("abm",                  "ABM Industries",            "abm.com",               "Facility Services",       "NA"),
    ("comfort-systems",      "Comfort Systems USA",       "comfortsystemsusa.com", "Field Services",          "NA"),
    ("rollins",              "Rollins",                   "rollins.com",           "Pest Control",            "NA"),
    ("terminix",             "Terminix",                  "terminix.com",          "Pest Control",            "NA"),
    ("brightview",           "BrightView",                "brightview.com",        "Landscaping",             "NA"),
    ("trugreen",             "TruGreen",                  "trugreen.com",          "Landscaping",             "NA"),
    # Grocery & specialty retail (private delivery fleets)
    ("kroger",               "Kroger",                    "kroger.com",            "Grocery",                 "NA"),
    ("sherwin-williams",     "Sherwin-Williams",          "sherwin-williams.com",  "Specialty Retail",        "NA"),
]

PRODUCTS: list[dict[str, Any]] = [
    # Platform tiers — per-vehicle telematics subscription
    {"sku": "fleetwise-starter",        "name": "Fleetwise Starter",                 "category": "platform", "tier": "starter",    "list_price_usd": 199},
    {"sku": "fleetwise-pro",            "name": "Fleetwise Pro",                     "category": "platform", "tier": "pro",        "list_price_usd": 999},
    {"sku": "fleetwise-enterprise",     "name": "Fleetwise Enterprise",              "category": "platform", "tier": "enterprise", "list_price_usd": 4999},
    {"sku": "fleetwise-asset-tracking", "name": "Fleetwise Asset Tracking",          "category": "platform", "tier": "enterprise", "list_price_usd": 3500},
    # Add-ons — modules that drive attach / whitespace plays
    {"sku": "maintenance",            "name": "Maintenance & Service Scheduling",    "category": "addon",    "tier": None,         "list_price_usd": 299},
    {"sku": "eld-compliance",         "name": "ELD & Hours-of-Service Compliance",   "category": "addon",    "tier": None,         "list_price_usd": 499},
    {"sku": "dashcam-ai",             "name": "AI Dashcam & Safety",                 "category": "addon",    "tier": None,         "list_price_usd": 799},
    {"sku": "tpms",                   "name": "Tire & TPMS Monitoring",              "category": "addon",    "tier": None,         "list_price_usd": 149},
    {"sku": "fuel-cards",             "name": "Fuel Card Program",                   "category": "addon",    "tier": None,         "list_price_usd": 199},
    {"sku": "driver-app",             "name": "Driver Mobile App",                   "category": "addon",    "tier": None,         "list_price_usd": 99},
    {"sku": "asset-tags",             "name": "Trailer & Asset Tags",                "category": "addon",    "tier": None,         "list_price_usd": 249},
    {"sku": "cold-chain",             "name": "Cold Chain Temperature Monitoring",   "category": "addon",    "tier": None,         "list_price_usd": 899},
    {"sku": "route-optimization",     "name": "Route Optimization",                  "category": "addon",    "tier": None,         "list_price_usd": 299},
    {"sku": "ifta-fuel-tax",          "name": "IFTA Fuel Tax Reporting",             "category": "addon",    "tier": None,         "list_price_usd": 199},
    {"sku": "ev-fleet",               "name": "EV Fleet & Charging Management",      "category": "addon",    "tier": None,         "list_price_usd": 1499},
    # Support
    {"sku": "support-priority",       "name": "Priority Support",                    "category": "support",  "tier": "premium",    "list_price_usd": 999},
    {"sku": "support-247",            "name": "24x7 Fleet Support",                  "category": "support",  "tier": "enterprise", "list_price_usd": 3999},
    # Services
    {"sku": "installation-pack",      "name": "Hardware Installation & Onboarding",  "category": "services", "tier": None,         "list_price_usd": 2500},
    {"sku": "pro-services",           "name": "Fleet Consulting Services",           "category": "services", "tier": None,         "list_price_usd": 4500},
    {"sku": "beta-access",            "name": "Beta Feature Access",                 "category": "services", "tier": None,         "list_price_usd": 0},
]

LAUNCH_DATES = {
    "platform": date(2022, 1, 1),
    "addon":    date(2022, 6, 1),
    "support":  date(2022, 1, 1),
    "services": date(2023, 1, 1),
}

CONTACT_ROLES = [
    ("champion",       ["Fleet Manager", "Fleet Operations Manager", "Maintenance Manager", "Safety Manager"], False, True),
    ("economic_buyer", ["VP Operations", "VP Fleet", "Director of Logistics", "Director of Fleet"],            True,  False),
    ("technical",      ["Telematics Administrator", "Dispatch Lead", "Shop Foreman", "Compliance Specialist"], False, False),
    ("executive",      ["Chief Operating Officer", "SVP Supply Chain", "EVP Operations"],                      True,  False),
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

    platform = PLATFORM_PREF[size_class]
    if rng.random() < MISTIER_PROB:
        platform = PLATFORM_DOWNGRADE[platform]

    chosen = [platform]
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
        # "new" accounts only have a short, recent history.
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
    """Derive arr_usd / arr_band / health_score / seats_total from real usage."""
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
    sample = next(a for a in data["accounts"] if a["account_id"] == "sysco")
    print("sample (sysco):", {k: sample[k] for k in
          ("size_class", "employee_count_band", "trajectory",
           "arr_band_usd", "arr_usd", "health_score", "owner_csm")})
