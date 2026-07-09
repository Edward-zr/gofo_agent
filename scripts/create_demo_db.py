"""Create a fresh GOFO demo SQLite database for SQL capability testing."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# Resolve paths from this script location so execution works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "gofo_demo.db"

# ---------------------------------------------------------------------------
# Demo data constants
# ---------------------------------------------------------------------------
HUBS = [
    "Los Angeles Hub",
    "Chicago Hub",
    "New York Hub",
    "Dallas Hub",
    "Atlanta Hub",
]

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Avery", "Quinn",
    "Jamie", "Drew", "Blake", "Cameron", "Logan", "Parker", "Reese",
]

LAST_NAMES = [
    "Chen", "Patel", "Johnson", "Garcia", "Nguyen", "Kim", "Brown", "Martinez",
    "Wilson", "Anderson", "Thomas", "Lee", "Clark", "Lewis", "Walker",
]

CUSTOMER_NAMES = [
    "Amazon", "Walmart", "Target", "Costco", "Home Depot", "Best Buy",
    "Nike", "Apple", "Samsung", "Wayfair", "Chewy", "Shopify Merchant",
    "TikTok Shop Seller", "Shein Partner", "Temul Logistics", "FedEx Ground Partner",
    "UPS Supply Chain", "DHL eCommerce", "Zara Distribution", "Uniqlo Retail",
    "Gap Inc.", "Nordstrom", "Macy's", "Kohl's", "CVS Pharmacy",
    "Walgreens", "Starbucks Roasting", "Peloton", "Lululemon", "Patagonia",
]

CITIES_STATES = [
    ("Los Angeles", "CA"), ("San Francisco", "CA"), ("San Diego", "CA"),
    ("Chicago", "IL"), ("Aurora", "IL"), ("Naperville", "IL"),
    ("New York", "NY"), ("Brooklyn", "NY"), ("Buffalo", "NY"),
    ("Dallas", "TX"), ("Houston", "TX"), ("Austin", "TX"),
    ("Atlanta", "GA"), ("Savannah", "GA"), ("Marietta", "GA"),
    ("Seattle", "WA"), ("Portland", "OR"), ("Phoenix", "AZ"),
    ("Denver", "CO"), ("Miami", "FL"), ("Orlando", "FL"),
    ("Boston", "MA"), ("Philadelphia", "PA"), ("Detroit", "MI"),
    ("Minneapolis", "MN"), ("Charlotte", "NC"), ("Nashville", "TN"),
    ("Columbus", "OH"), ("Indianapolis", "IN"), ("Kansas City", "MO"),
    ("Las Vegas", "NV"), ("Salt Lake City", "UT"), ("Raleigh", "NC"),
    ("Tampa", "FL"), ("San Antonio", "TX"), ("Baltimore", "MD"),
    ("Milwaukee", "WI"), ("St. Louis", "MO"), ("Cincinnati", "OH"),
    ("Pittsburgh", "PA"), ("Richmond", "VA"), ("Boise", "ID"),
    ("Albuquerque", "NM"), ("Omaha", "NE"), ("Tucson", "AZ"),
    ("Honolulu", "HI"), ("Anchorage", "AK"), ("Newark", "NJ"),
    ("Jersey City", "NJ"), ("Arlington", "VA"),
]

EXCEPTION_REASONS = [
    "Driver unavailable",
    "Address issue",
    "Customer unavailable",
    "Weather",
    "Vehicle issue",
    "Capacity exceeded",
]

def _build_status_pool(size: int) -> list[str]:
    """Build a pickup status list matching the target distribution."""
    completed = int(size * 0.85)
    delayed = int(size * 0.10)
    failed = size - completed - delayed
    pool = (["Completed"] * completed) + (["Delayed"] * delayed) + (["Failed"] * failed)
    random.shuffle(pool)
    return pool


def _random_pickup_dates(count: int, days_back: int = 30) -> list[str]:
    """Generate pickup dates spread across the recent date window."""
    today = date.today()
    start = today - timedelta(days=days_back - 1)
    return [
        (start + timedelta(days=random.randint(0, days_back - 1))).isoformat()
        for _ in range(count)
    ]


def create_schema(connection: sqlite3.Connection) -> None:
    """Create all analytics tables with primary and foreign keys."""
    connection.executescript(
        """
        CREATE TABLE drivers (
            driver_id INTEGER PRIMARY KEY,
            driver_name TEXT NOT NULL,
            hub TEXT NOT NULL
        );

        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL
        );

        CREATE TABLE addresses (
            address_id INTEGER PRIMARY KEY,
            city TEXT NOT NULL,
            state TEXT NOT NULL
        );

        CREATE TABLE pickups (
            pickup_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            address_id INTEGER NOT NULL,
            pickup_date TEXT NOT NULL,
            package_count INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('Completed', 'Delayed', 'Failed')),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (driver_id) REFERENCES drivers(driver_id),
            FOREIGN KEY (address_id) REFERENCES addresses(address_id)
        );

        CREATE TABLE exceptions (
            exception_id INTEGER PRIMARY KEY,
            pickup_id INTEGER NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (pickup_id) REFERENCES pickups(pickup_id)
        );
        """
    )


def seed_drivers(connection: sqlite3.Connection, count: int = 20) -> None:
    """Insert demo driver records assigned to operational hubs."""
    rows = []
    for driver_id in range(1, count + 1):
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        hub = random.choice(HUBS)
        rows.append((driver_id, name, hub))
    connection.executemany(
        "INSERT INTO drivers (driver_id, driver_name, hub) VALUES (?, ?, ?)",
        rows,
    )


def seed_customers(connection: sqlite3.Connection, count: int = 30) -> None:
    """Insert demo customer records."""
    rows = [(index, CUSTOMER_NAMES[index - 1]) for index in range(1, count + 1)]
    connection.executemany(
        "INSERT INTO customers (customer_id, customer_name) VALUES (?, ?)",
        rows,
    )


def seed_addresses(connection: sqlite3.Connection, count: int = 50) -> None:
    """Insert demo address records across US cities."""
    selected = random.sample(CITIES_STATES, count)
    rows = [(index, city, state) for index, (city, state) in enumerate(selected, start=1)]
    connection.executemany(
        "INSERT INTO addresses (address_id, city, state) VALUES (?, ?, ?)",
        rows,
    )


def seed_pickups(connection: sqlite3.Connection, count: int = 500) -> list[int]:
    """Insert demo pickup records distributed across the last 30 days."""
    statuses = _build_status_pool(count)
    pickup_dates = _random_pickup_dates(count)
    rows = []
    for pickup_id in range(1, count + 1):
        rows.append(
            (
                pickup_id,
                random.randint(1, 30),
                random.randint(1, 20),
                random.randint(1, 50),
                pickup_dates[pickup_id - 1],
                random.randint(1, 300),
                statuses[pickup_id - 1],
            )
        )
    connection.executemany(
        """
        INSERT INTO pickups (
            pickup_id, customer_id, driver_id, address_id,
            pickup_date, package_count, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return list(range(1, count + 1))


def seed_exceptions(connection: sqlite3.Connection, pickup_ids: list[int], count: int = 60) -> None:
    """Insert demo exception records linked to non-completed pickups when possible."""
    non_completed = connection.execute(
        "SELECT pickup_id, pickup_date FROM pickups WHERE status IN ('Delayed', 'Failed')"
    ).fetchall()
    candidate_ids = [row[0] for row in non_completed]
    if len(candidate_ids) < count:
        candidate_ids = pickup_ids

    selected_pickups = random.choices(candidate_ids, k=count)
    rows = []
    for exception_id, pickup_id in enumerate(selected_pickups, start=1):
        pickup_date = connection.execute(
            "SELECT pickup_date FROM pickups WHERE pickup_id = ?",
            (pickup_id,),
        ).fetchone()[0]
        created_at = f"{pickup_date} {random.randint(8, 20):02d}:{random.randint(0, 59):02d}:00"
        rows.append(
            (
                exception_id,
                pickup_id,
                random.choice(EXCEPTION_REASONS),
                created_at,
            )
        )
    connection.executemany(
        "INSERT INTO exceptions (exception_id, pickup_id, reason, created_at) VALUES (?, ?, ?, ?)",
        rows,
    )


def print_summary(connection: sqlite3.Connection) -> None:
    """Print row counts for each table after database creation."""
    counts = {
        "Drivers": connection.execute("SELECT COUNT(*) FROM drivers").fetchone()[0],
        "Customers": connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0],
        "Addresses": connection.execute("SELECT COUNT(*) FROM addresses").fetchone()[0],
        "Pickups": connection.execute("SELECT COUNT(*) FROM pickups").fetchone()[0],
        "Exceptions": connection.execute("SELECT COUNT(*) FROM exceptions").fetchone()[0],
    }

    print("Database created successfully.")
    print(f"Location: {DATABASE_PATH.relative_to(PROJECT_ROOT)}")
    for label, value in counts.items():
        print(f"{label}: {value}")


def main() -> None:
    """Delete any existing demo database and recreate it from scratch."""
    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    connection = sqlite3.connect(DATABASE_PATH)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        create_schema(connection)
        seed_drivers(connection)
        seed_customers(connection)
        seed_addresses(connection)
        pickup_ids = seed_pickups(connection)
        seed_exceptions(connection, pickup_ids)
        connection.commit()
        print_summary(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
