"""Database initialization, schema management, and data access layer for
Smart Garage & Spare Parts Assistance System for Royal Enfield Thunderbird 350X.
Uses SQLite with parameterized queries and Werkzeug password hashing.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from werkzeug.security import check_password_hash, generate_password_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Backend" / "data"
DB_PATH = DATA_DIR / "thunderbird_garage.db"
DATASET_PATH = PROJECT_ROOT / "Dataset" / "royal_enfield_thunderbird_350x_parts_dataset.csv"


def get_db_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Creates database tables if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                phone TEXT,
                city TEXT DEFAULT 'Pune',
                role TEXT NOT NULL CHECK(role IN ('user', 'garage')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Garages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS garages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE,
                garage_name TEXT NOT NULL,
                garage_type TEXT NOT NULL,
                city TEXT NOT NULL,
                contact TEXT NOT NULL,
                address TEXT NOT NULL,
                rating REAL DEFAULT 4.5,
                speciality TEXT DEFAULT 'Royal Enfield Thunderbird 350X Specialist',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
            );
        """)

        # Spare parts inventory (seeded from academic dataset)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spare_parts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bike_model TEXT NOT NULL,
                manufacturing_year INTEGER NOT NULL,
                part_name TEXT NOT NULL,
                part_category TEXT NOT NULL,
                availability_status TEXT NOT NULL,
                estimated_wait_days INTEGER NOT NULL,
                estimated_price INTEGER NOT NULL,
                supplier_type TEXT NOT NULL,
                city TEXT NOT NULL,
                garage_type TEXT NOT NULL,
                part_condition TEXT NOT NULL,
                service_priority TEXT NOT NULL,
                record_date TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Service requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS service_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                garage_id INTEGER,
                user_name TEXT NOT NULL,
                user_phone TEXT NOT NULL,
                bike_model TEXT NOT NULL,
                part_name TEXT NOT NULL,
                part_category TEXT,
                issue_description TEXT,
                city TEXT NOT NULL,
                priority TEXT NOT NULL CHECK(priority IN ('Low', 'Medium', 'High', 'Critical')),
                estimated_price REAL DEFAULT 0,
                estimated_wait_days INTEGER DEFAULT 0,
                status TEXT NOT NULL CHECK(status IN ('Pending', 'Accepted', 'In Progress', 'Completed', 'Cancelled')),
                garage_notes TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (garage_id) REFERENCES garages (id) ON DELETE SET NULL
            );
        """)

        # Prediction history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                part_name TEXT NOT NULL,
                part_category TEXT NOT NULL,
                city TEXT NOT NULL,
                predicted_availability TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                predicted_wait_days INTEGER NOT NULL,
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
        """)

        conn.commit()


def seed_db() -> None:
    """Seeds default demo accounts, dataset spare parts, and demo service requests safely and idempotently."""
    init_db()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Seed demo user
        demo_user_pw = generate_password_hash("password123")
        cursor.execute("SELECT id FROM users WHERE username = ?", ("user_demo",))
        user_row = cursor.fetchone()
        if not user_row:
            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, full_name, phone, city, role)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "user_demo",
                    "user@thunderbird.local",
                    demo_user_pw,
                    "Rajvardhan Patil",
                    "+91 98765 43210",
                    "Pune",
                    "user",
                ),
            )
            user_id = cursor.lastrowid
        else:
            user_id = user_row["id"]

        # 2. Seed demo garage account & garage profiles
        demo_garage_pw = generate_password_hash("garage123")
        cursor.execute("SELECT id FROM users WHERE username = ?", ("garage_demo",))
        garage_user_row = cursor.fetchone()
        if not garage_user_row:
            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, full_name, phone, city, role)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "garage_demo",
                    "garage@thunderbird.local",
                    demo_garage_pw,
                    "Royal Enfield Specialist Pune",
                    "+91 91234 56789",
                    "Pune",
                    "garage",
                ),
            )
            garage_user_id = cursor.lastrowid
        else:
            garage_user_id = garage_user_row["id"]

        # Ensure primary demo garage profile is linked
        cursor.execute("SELECT id FROM garages WHERE user_id = ?", (garage_user_id,))
        primary_garage = cursor.fetchone()
        if not primary_garage:
            cursor.execute(
                """
                INSERT INTO garages (user_id, garage_name, garage_type, city, contact, address, rating, speciality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    garage_user_id,
                    "Bullet Masters - RE Specialist Garage",
                    "Royal Enfield Specialist",
                    "Pune",
                    "+91 91234 56789",
                    "Shop 14, FC Road, Shivajinagar, Pune, MH",
                    4.8,
                    "Thunderbird 350X EFI & Dual Channel ABS Expert",
                ),
            )
            primary_garage_id = cursor.lastrowid
        else:
            primary_garage_id = primary_garage["id"]

        # Seed other representative garages across cities
        sample_garages = [
            ("Mumbai Royal Enfield Care", "Authorized Service Center", "Mumbai", "+91 98220 11223", "Andheri West, Link Road, Mumbai", 4.7, "OEM Spares & Engine Overhauls"),
            ("Deccan Bike Craft", "Multi-brand Bike Garage", "Pune", "+91 98220 33445", "Kothrud Depot, Paud Road, Pune", 4.5, "Custom Cruiser Modifications & Electricals"),
            ("Nashik Riders Garage", "Royal Enfield Specialist", "Nashik", "+91 98220 55667", "College Road, Gangapur Naka, Nashik", 4.6, "Transmission & Clutch Specialists"),
            ("Kolhapur Cruiser Hub", "Local Motorcycle Garage", "Kolhapur", "+91 98220 77889", "Shahupuri, Station Road, Kolhapur", 4.4, "Vintage & Discontinued RE Models"),
            ("Aurangabad Auto Spares & Garage", "Authorized Service Center", "Aurangabad", "+91 98220 99001", "Cidco Cannaught Place, Aurangabad", 4.6, "Suspension & Brake Line Servicing"),
            ("Satara Bullet Service Station", "Royal Enfield Specialist", "Satara", "+91 98221 11223", "Powai Naka, Satara", 4.5, "Standard & Custom Thunderbird Tuning"),
        ]

        for g_name, g_type, g_city, g_contact, g_addr, g_rating, g_spec in sample_garages:
            cursor.execute("SELECT id FROM garages WHERE garage_name = ?", (g_name,))
            if not cursor.fetchone():
                cursor.execute(
                    """
                    INSERT INTO garages (garage_name, garage_type, city, contact, address, rating, speciality)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (g_name, g_type, g_city, g_contact, g_addr, g_rating, g_spec),
                )

        # 3. Seed spare parts from CSV if empty
        cursor.execute("SELECT COUNT(*) AS cnt FROM spare_parts")
        parts_count = cursor.fetchone()["cnt"]
        if parts_count == 0 and DATASET_PATH.exists():
            df = pd.read_csv(DATASET_PATH)
            for _, row in df.iterrows():
                cursor.execute(
                    """
                    INSERT INTO spare_parts (
                        bike_model, manufacturing_year, part_name, part_category,
                        availability_status, estimated_wait_days, estimated_price,
                        supplier_type, city, garage_type, part_condition, service_priority,
                        record_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(row.get("Bike model", "Royal Enfield Thunderbird 350X")),
                        int(row.get("Manufacturing year", 2019)),
                        str(row.get("Part name", "Unknown")),
                        str(row.get("Part category", "General")),
                        str(row.get("Availability status", "Available")),
                        int(row.get("Estimated wait days", 0)),
                        int(row.get("Estimated price (INR)", 0)),
                        str(row.get("Supplier type", "Independent Distributor")),
                        str(row.get("City", "Pune")),
                        str(row.get("Garage type", "Royal Enfield Specialist")),
                        str(row.get("Part condition", "New")),
                        str(row.get("Service priority", "Medium")),
                        str(row.get("Record date", "2025-01-01")),
                    ),
                )

        # 4. Seed demo service requests if none exist
        cursor.execute("SELECT COUNT(*) AS cnt FROM service_requests")
        if cursor.fetchone()["cnt"] == 0:
            demo_requests = [
                (
                    "SR-2026-001",
                    user_id,
                    primary_garage_id,
                    "Rajvardhan Patil",
                    "+91 98765 43210",
                    "Royal Enfield Thunderbird 350X",
                    "Clutch Cable",
                    "Control",
                    "Clutch cable friction and free-play issue causing heavy lever pull during city commute.",
                    "Pune",
                    "High",
                    280.0,
                    2,
                    "In Progress",
                    "Genuine OEM clutch cable sourced; replacement scheduled for 3 PM.",
                ),
                (
                    "SR-2026-002",
                    user_id,
                    primary_garage_id,
                    "Rajvardhan Patil",
                    "+91 98765 43210",
                    "Royal Enfield Thunderbird 350X",
                    "Front Disc Brake Pads",
                    "Braking",
                    "Front disc brake squeaking noise and reduced bite upon sudden stops.",
                    "Pune",
                    "Medium",
                    650.0,
                    1,
                    "Accepted",
                    "Inspected rotor wear; brake pads cleaning & replacement accepted.",
                ),
                (
                    "SR-2026-003",
                    user_id,
                    None,
                    "Rajvardhan Patil",
                    "+91 98765 43210",
                    "Royal Enfield Thunderbird 350X",
                    "Fuel Pump Motor Assembly",
                    "Engine",
                    "Low priming sound from tank, sporadic hesitation at 3000 RPM on highway rides.",
                    "Pune",
                    "Critical",
                    4800.0,
                    7,
                    "Pending",
                    "",
                ),
            ]

            for req in demo_requests:
                cursor.execute(
                    """
                    INSERT INTO service_requests (
                        request_id, user_id, garage_id, user_name, user_phone,
                        bike_model, part_name, part_category, issue_description,
                        city, priority, estimated_price, estimated_wait_days,
                        status, garage_notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    req,
                )

        conn.commit()


# Helper query functions

def authenticate_user(username: str, password: str, expected_role: Optional[str] = None) -> Optional[dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, username))
        user = cursor.fetchone()
        if user and check_password_hash(user["password_hash"], password):
            if expected_role and user["role"] != expected_role:
                return None
            return dict(user)
    return None


def register_user(username: str, email: str, password: str, full_name: str, phone: str, city: str, role: str = "user") -> tuple[bool, str, Optional[int]]:
    if role not in ("user", "garage"):
        return False, "Invalid role.", None
    if len(password) < 6:
        return False, "Password must be at least 6 characters long.", None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            return False, "Username or email is already registered.", None

        pw_hash = generate_password_hash(password)
        cursor.execute(
            """
            INSERT INTO users (username, email, password_hash, full_name, phone, city, role)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (username, email, pw_hash, full_name, phone, city, role),
        )
        new_id = cursor.lastrowid

        if role == "garage":
            cursor.execute(
                """
                INSERT INTO garages (user_id, garage_name, garage_type, city, contact, address, rating)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (new_id, full_name, "Royal Enfield Specialist", city, phone, f"{city} Center", 4.5),
            )
        conn.commit()
        return True, "Registration successful.", new_id


def get_user_by_id(user_id: int) -> Optional[dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, full_name, phone, city, role, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_garage_by_user_id(user_id: int) -> Optional[dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM garages WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_garages(city: Optional[str] = None) -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if city:
            cursor.execute("SELECT * FROM garages WHERE city LIKE ? ORDER BY rating DESC", (f"%{city}%",))
        else:
            cursor.execute("SELECT * FROM garages ORDER BY rating DESC")
        return [dict(row) for row in cursor.fetchall()]


def search_spare_parts(
    part_name: Optional[str] = None,
    category: Optional[str] = None,
    availability: Optional[str] = None,
    city: Optional[str] = None,
    condition: Optional[str] = None,
    max_price: Optional[float] = None,
    max_wait_days: Optional[int] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM spare_parts WHERE 1=1"
    params: list[Any] = []

    if part_name:
        query += " AND part_name LIKE ?"
        params.append(f"%{part_name}%")
    if category:
        query += " AND part_category = ?"
        params.append(category)
    if availability:
        query += " AND availability_status = ?"
        params.append(availability)
    if city:
        query += " AND city LIKE ?"
        params.append(f"%{city}%")
    if condition:
        query += " AND part_condition = ?"
        params.append(condition)
    if max_price is not None and max_price > 0:
        query += " AND estimated_price <= ?"
        params.append(max_price)
    if max_wait_days is not None and max_wait_days >= 0:
        query += " AND estimated_wait_days <= ?"
        params.append(max_wait_days)

    query += " ORDER BY id ASC LIMIT ?"
    params.append(limit)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]


def create_service_request(
    user_id: int,
    user_name: str,
    user_phone: str,
    bike_model: str,
    part_name: str,
    city: str,
    priority: str,
    issue_description: str = "",
    garage_id: Optional[int] = None,
    part_category: Optional[str] = None,
    estimated_price: float = 0.0,
    estimated_wait_days: int = 0,
) -> str:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Generate unique request id: SR-YYYY-XXXX
        year = datetime.now().year
        cursor.execute("SELECT COUNT(*) AS cnt FROM service_requests")
        count = cursor.fetchone()["cnt"] + 1
        req_id = f"SR-{year}-{count:04d}"

        cursor.execute(
            """
            INSERT INTO service_requests (
                request_id, user_id, garage_id, user_name, user_phone,
                bike_model, part_name, part_category, issue_description,
                city, priority, estimated_price, estimated_wait_days,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Pending')
            """,
            (
                req_id,
                user_id,
                garage_id,
                user_name,
                user_phone,
                bike_model,
                part_name,
                part_category,
                issue_description,
                city,
                priority,
                estimated_price,
                estimated_wait_days,
            ),
        )
        conn.commit()
        return req_id


def get_user_service_requests(user_id: int) -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT sr.*, g.garage_name, g.contact AS garage_contact
            FROM service_requests sr
            LEFT JOIN garages g ON sr.garage_id = g.id
            WHERE sr.user_id = ?
            ORDER BY sr.id DESC
            """,
            (user_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_garage_service_requests(garage_id: int, city: Optional[str] = None) -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Returns requests assigned to this garage OR pending in the same city
        if city:
            cursor.execute(
                """
                SELECT sr.*, u.email AS user_email
                FROM service_requests sr
                JOIN users u ON sr.user_id = u.id
                WHERE sr.garage_id = ? OR (sr.garage_id IS NULL AND sr.city LIKE ?)
                ORDER BY CASE sr.status
                    WHEN 'Pending' THEN 1
                    WHEN 'Accepted' THEN 2
                    WHEN 'In Progress' THEN 3
                    ELSE 4 END, sr.id DESC
                """,
                (garage_id, f"%{city}%"),
            )
        else:
            cursor.execute(
                """
                SELECT sr.*, u.email AS user_email
                FROM service_requests sr
                JOIN users u ON sr.user_id = u.id
                WHERE sr.garage_id = ? OR sr.garage_id IS NULL
                ORDER BY sr.id DESC
                """,
                (garage_id,),
            )
        return [dict(row) for row in cursor.fetchall()]


def update_service_request_status(request_id: str, garage_id: int, new_status: str, notes: str = "") -> bool:
    valid_statuses = ("Pending", "Accepted", "In Progress", "Completed", "Cancelled")
    if new_status not in valid_statuses:
        return False

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Verify request exists and is either assigned to this garage or unassigned
        cursor.execute("SELECT id, garage_id FROM service_requests WHERE request_id = ?", (request_id,))
        row = cursor.fetchone()
        if not row:
            return False

        current_garage_id = row["garage_id"]
        if current_garage_id is not None and current_garage_id != garage_id:
            # Not allowed to edit requests assigned to another garage
            return False

        cursor.execute(
            """
            UPDATE service_requests
            SET status = ?, garage_id = ?, garage_notes = ?, updated_at = CURRENT_TIMESTAMP
            WHERE request_id = ?
            """,
            (new_status, garage_id, notes, request_id),
        )
        conn.commit()
        return True


def log_prediction(
    user_id: Optional[int],
    part_name: str,
    part_category: str,
    city: str,
    availability: str,
    price: float,
    wait_days: int,
    confidence: float = 0.0,
) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_history (
                user_id, part_name, part_category, city,
                predicted_availability, predicted_price, predicted_wait_days, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, part_name, part_category, city, availability, price, wait_days, confidence),
        )
        conn.commit()
