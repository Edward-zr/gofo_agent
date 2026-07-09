"""SQLite schema documentation for GOFO analytics SQL planning."""

SCHEMA = """
Database Engine
---------------
SQLite

Tables
------

drivers
-------
driver_id INTEGER PRIMARY KEY
driver_name TEXT NOT NULL
hub TEXT NOT NULL

customers
---------
customer_id INTEGER PRIMARY KEY
customer_name TEXT NOT NULL

addresses
---------
address_id INTEGER PRIMARY KEY
city TEXT NOT NULL
state TEXT NOT NULL

pickups
-------
pickup_id INTEGER PRIMARY KEY
customer_id INTEGER NOT NULL REFERENCES customers(customer_id)
driver_id INTEGER NOT NULL REFERENCES drivers(driver_id)
address_id INTEGER NOT NULL REFERENCES addresses(address_id)
pickup_date TEXT NOT NULL
package_count INTEGER NOT NULL
status TEXT NOT NULL CHECK(status IN ('Completed', 'Delayed', 'Failed'))

exceptions
----------
exception_id INTEGER PRIMARY KEY
pickup_id INTEGER NOT NULL REFERENCES pickups(pickup_id)
reason TEXT NOT NULL
created_at TEXT NOT NULL

Relationships
-------------
pickups.customer_id -> customers.customer_id
pickups.driver_id -> drivers.driver_id
pickups.address_id -> addresses.address_id
exceptions.pickup_id -> pickups.pickup_id

Notes
-----
pickup_date stores ISO dates such as '2026-06-28'.
status values are exactly: Completed, Delayed, Failed.
exception reason examples include Driver unavailable, Address issue,
Customer unavailable, Weather, Vehicle issue, Capacity exceeded.
"""
