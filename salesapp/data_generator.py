"""Synthetic (and deliberately messy) e-commerce order data.

The generator produces one row per order line. After the clean data is built,
realistic dirt is injected so the cleaning pipeline has real work to do:
duplicate lines, mixed date formats, currency-formatted prices, inconsistent
text casing, missing values and invalid quantities. Three flash-sale days are
planted so the anomaly detector has something real to find.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATALOG: dict[str, list[tuple[str, int]]] = {
    "Electronics": [
        ("Wireless Earbuds", 2499), ("Bluetooth Speaker", 3299), ("Smartwatch", 7999),
        ("Power Bank 20000mAh", 1799), ("Mechanical Keyboard", 4499), ("Gaming Mouse", 1999),
        ("4K Action Camera", 11999), ("Tablet 10in", 21999),
    ],
    "Fashion": [
        ("Denim Jacket", 2799), ("Running Shoes", 3499), ("Cotton Kurta", 1299),
        ("Leather Wallet", 899), ("Sunglasses", 1599), ("Hoodie", 1899),
        ("Formal Shirt", 1499), ("Backpack", 2199),
    ],
    "Home & Kitchen": [
        ("Air Fryer", 6499), ("Mixer Grinder", 3999), ("Bedsheet Set", 1799),
        ("Water Bottle Set", 699), ("LED Desk Lamp", 1199), ("Non-stick Cookware", 2999),
        ("Vacuum Cleaner", 8999), ("Wall Clock", 849),
    ],
    "Beauty": [
        ("Face Serum", 899), ("Sunscreen SPF 50", 499), ("Hair Dryer", 1799),
        ("Perfume 100ml", 1999), ("Beard Trimmer", 1499), ("Lip Care Kit", 349),
        ("Face Wash Combo", 449), ("Skincare Gift Set", 2499),
    ],
    "Books": [
        ("Python Crash Course", 699), ("Atomic Habits", 499), ("Data Science Handbook", 899),
        ("Engineering Maths", 599), ("Fiction Bestseller", 349), ("Comics Box Set", 999),
        ("UPSC Guide", 799), ("Notebook Pack", 249),
    ],
    "Sports": [
        ("Yoga Mat", 899), ("Dumbbell Set", 2999), ("Football", 1099), ("Cricket Bat", 3499),
        ("Badminton Racket", 1799), ("Fitness Band", 2499), ("Cycling Helmet", 1599),
        ("Skipping Rope", 249),
    ],
}

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Arjun", "Ishaan", "Rohan", "Kabir", "Siddharth",
               "Ananya", "Diya", "Isha", "Kavya", "Meera", "Priya", "Riya", "Saanvi",
               "Neha", "Pooja", "Rahul", "Karan", "Sneha", "Tanvi", "Manish", "Nikhil"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Patel", "Reddy", "Iyer", "Nair",
              "Mehta", "Joshi", "Kapoor", "Mishra", "Tiwari", "Das", "Bose", "Khan"]

REGIONS = ["North", "South", "East", "West", "Central"]
REGION_P = [0.28, 0.24, 0.14, 0.24, 0.10]
CHANNELS = ["Web", "Mobile App", "Marketplace"]
CHANNEL_P = [0.30, 0.50, 0.20]
PAYMENTS = ["UPI", "Credit Card", "Debit Card", "COD", "Net Banking"]
PAYMENT_P = [0.45, 0.20, 0.15, 0.12, 0.08]
STATUSES = ["Delivered", "Returned", "Cancelled"]
STATUS_P = [0.90, 0.05, 0.05]
FLASH_SALE_DATES = ["2024-10-30", "2025-08-15", "2026-01-26"]

# tier -> (mean days between orders, mean customer lifetime in days)
TIERS = {"casual": (110.0, 100.0), "regular": (50.0, 450.0), "loyal": (24.0, 800.0)}
# index 1..12 = seasonal keep-probability by calendar month (festive season is stronger)
MONTH_WEIGHT = np.array([0, .75, .75, .80, .80, .85, .85, .90, .95, 1.0, 1.0, 1.0, 1.0])

RAW_COLUMNS = [
    "line_id", "order_id", "order_date", "customer_id", "customer_name", "product_id",
    "product_name", "category", "region", "channel", "quantity", "unit_price",
    "discount_pct", "payment_method", "order_status",
]


def _build_catalog(rng: np.random.Generator) -> pd.DataFrame:
    rows, pid = [], 1
    for category, items in CATALOG.items():
        for name, price in items:
            rows.append((f"P{pid:03d}", name, category, float(price)))
            pid += 1
    catalog = pd.DataFrame(rows, columns=["product_id", "product_name", "category", "base_price"])
    weights = 1.0 / np.arange(1, len(catalog) + 1) ** 0.9  # a few hero products, long tail
    catalog["popularity"] = rng.permutation(weights / weights.sum())
    return catalog


def _build_customers(rng: np.random.Generator, n: int, start: pd.Timestamp, span_days: int) -> pd.DataFrame:
    first = rng.choice(FIRST_NAMES, n)
    last = rng.choice(LAST_NAMES, n)
    offsets = (span_days * rng.beta(1.4, 1.0, n)).astype(int)  # steady customer-base growth
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "customer_name": np.char.add(np.char.add(first, " "), last),
        "signup_date": start + pd.to_timedelta(offsets, unit="D"),
        "region": rng.choice(REGIONS, n, p=REGION_P),
        "tier": rng.choice(list(TIERS), n, p=[0.5, 0.35, 0.15]),
    })


def _build_orders(rng: np.random.Generator, customers: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    ids, dates, regions = [], [], []
    for cid, signup, region, tier in zip(
        customers["customer_id"], customers["signup_date"], customers["region"], customers["tier"]
    ):
        gap, lifetime = TIERS[tier]
        horizon = (end - signup).days
        active_days = min(horizon, rng.exponential(lifetime))
        m = int(1.5 * horizon / gap) + 10
        offsets = np.concatenate([[0.0], np.cumsum(rng.exponential(gap, m))])
        offsets = offsets[offsets <= active_days]  # purchasing stops when the customer churns
        n = len(offsets)
        order_dates = (signup + pd.to_timedelta(offsets, unit="D")).floor("D")
        keep = rng.random(n) < MONTH_WEIGHT[order_dates.month.to_numpy()]
        keep[0] = True  # a customer's first order is always kept (cohorts need it)
        keep &= np.asarray(order_dates <= end)
        k = int(keep.sum())
        ids.append(np.repeat(cid, k))
        dates.append(order_dates[keep].to_numpy())
        regions.append(np.repeat(region, k))
    return pd.DataFrame({
        "customer_id": np.concatenate(ids),
        "order_date": np.concatenate(dates),
        "region": np.concatenate(regions),
    })


def _flash_orders(rng: np.random.Generator, customers: pd.DataFrame, per_sale: int = 150) -> pd.DataFrame:
    parts = []
    for d in FLASH_SALE_DATES:
        day = pd.Timestamp(d)
        eligible = customers[customers["signup_date"] <= day]
        pick = eligible.sample(n=min(per_sale, len(eligible)), random_state=int(rng.integers(1_000_000_000)))
        parts.append(pd.DataFrame({
            "customer_id": pick["customer_id"].to_numpy(),
            "order_date": day,
            "region": pick["region"].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def _build_lines(rng: np.random.Generator, orders: pd.DataFrame, catalog: pd.DataFrame,
                 names: pd.Series) -> pd.DataFrame:
    n_orders = len(orders)
    orders = orders.assign(
        channel=rng.choice(CHANNELS, n_orders, p=CHANNEL_P),
        payment_method=rng.choice(PAYMENTS, n_orders, p=PAYMENT_P),
        order_status=rng.choice(STATUSES, n_orders, p=STATUS_P),
    )
    n_lines = rng.choice([1, 2, 3, 4], n_orders, p=[0.55, 0.25, 0.14, 0.06])
    lines = orders.loc[orders.index.repeat(n_lines)].reset_index(drop=True)
    n = len(lines)
    position = lines.groupby("order_id").cumcount().to_numpy()

    categories = catalog["category"].to_numpy()
    pidx = rng.choice(len(catalog), n, p=catalog["popularity"].to_numpy())
    # market-basket signal: later lines often come from the same category as the first line
    first_cat = (pd.Series(categories[pidx]).groupby(lines["order_id"].to_numpy()).transform("first").to_numpy())
    follow = (position > 0) & (rng.random(n) < 0.35)
    for cat in np.unique(categories):
        m = follow & (first_cat == cat)
        if m.any():
            pidx[m] = rng.choice(np.flatnonzero(categories == cat), int(m.sum()))
    prod = catalog.iloc[pidx].reset_index(drop=True)

    base = prod["base_price"].to_numpy()
    price = np.round(base * rng.normal(1.0, 0.02, n), 2)
    qty = np.clip(1 + rng.poisson(np.where(base > 5000, 0.05, 0.4)), 1, 5)
    is_flash = lines["order_date"].isin(pd.to_datetime(FLASH_SALE_DATES)).to_numpy()
    discount = np.where(
        is_flash,
        rng.choice([30, 40, 50], n),
        rng.choice([0, 5, 10, 15, 20], n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
    )
    return pd.DataFrame({
        "line_id": "L" + pd.Series(np.arange(1, n + 1)).astype(str).str.zfill(7),
        "order_id": lines["order_id"],
        "order_date": lines["order_date"],
        "customer_id": lines["customer_id"],
        "customer_name": lines["customer_id"].map(names),
        "product_id": prod["product_id"],
        "product_name": prod["product_name"],
        "category": prod["category"],
        "region": lines["region"],
        "channel": lines["channel"],
        "quantity": qty,
        "unit_price": price,
        "discount_pct": discount.astype(float),
        "payment_method": lines["payment_method"],
        "order_status": lines["order_status"],
    })


def _inject_dirt(rng: np.random.Generator, df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Turn the clean frame into the kind of export a real system would produce."""
    df = df.copy()
    n = len(df)

    iso = df["order_date"].dt.strftime("%Y-%m-%d")
    uk = df["order_date"].dt.strftime("%d/%m/%Y")
    df["order_date"] = iso.where(rng.random(n) >= 0.05, uk)

    price = df["unit_price"].astype(str)
    fancy = df["unit_price"].map(lambda v: f"₹{v:,.2f}")
    df["unit_price"] = price.where(rng.random(n) >= 0.03, fancy)

    region = df["region"]
    roll = rng.random(n)
    region = np.select(
        [roll < 0.03, roll < 0.06, roll < 0.08],
        [region.str.upper().to_numpy(), region.str.lower().to_numpy(), (" " + region + " ").to_numpy()],
        default=region.to_numpy(),
    )
    df["region"] = pd.Series(region).mask(rng.random(n) < 0.02)

    pay = df["payment_method"]
    roll = rng.random(n)
    df["payment_method"] = np.select(
        [roll < 0.04, roll < 0.08], [pay.str.lower().to_numpy(), pay.str.upper().to_numpy()],
        default=pay.to_numpy())

    df["discount_pct"] = df["discount_pct"].mask(rng.random(n) < 0.03)
    df["customer_name"] = df["customer_name"].mask(rng.random(n) < 0.01)

    bad = rng.random(n) < 0.003
    df["quantity"] = np.where(bad, rng.choice([0, -1], n), df["quantity"])

    dupes = df.sample(frac=0.01, random_state=seed)
    return pd.concat([df, dupes], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_sales_data(n_customers: int = 3000, seed: int = 42,
                        start: str = "2024-01-01", end: str = "2026-10-31") -> pd.DataFrame:
    """Return a messy raw order-line table (all values still typed loosely)."""
    rng = np.random.default_rng(seed)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    catalog = _build_catalog(rng)
    customers = _build_customers(rng, n_customers, start_ts, (end_ts - start_ts).days)

    orders = pd.concat(
        [_build_orders(rng, customers, end_ts), _flash_orders(rng, customers)], ignore_index=True
    ).sort_values(["order_date", "customer_id"], kind="stable").reset_index(drop=True)
    orders["order_id"] = "ORD-" + pd.Series(orders.index + 1).astype(str).str.zfill(6)

    names = customers.set_index("customer_id")["customer_name"]
    lines = _build_lines(rng, orders, catalog, names)
    return _inject_dirt(rng, lines, seed)[RAW_COLUMNS]


def write_sales_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path
