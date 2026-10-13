"""Comprehensive Automated Test Suite for
Royal Enfield Thunderbird 350X Smart Garage & Spare Parts Assistance System.
Runs with pytest and Flask test client.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import app, load_dataset, load_model  # type: ignore  # noqa: E402
from database import (  # type: ignore  # noqa: E402
    authenticate_user,
    create_service_request,
    get_db_connection,
    get_garage_service_requests,
    get_user_service_requests,
    init_db,
    search_spare_parts,
    seed_db,
    update_service_request_status,
)


@pytest.fixture(autouse=True)
def setup_database():
    """Ensure database is initialized and seeded before running tests."""
    init_db()
    seed_db()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client


# Phase 2: Dataset Verification

def test_dataset_loading():
    df = load_dataset()
    required = {
        "Bike model",
        "Manufacturing year",
        "Part name",
        "Part category",
        "Availability status",
        "Estimated wait days",
        "Estimated price (INR)",
        "Supplier type",
        "City",
        "Garage type",
        "Part condition",
        "Service priority",
        "Record date",
    }
    assert required.issubset(set(df.columns)), "Dataset missing required schema columns."
    assert len(df) == 500, f"Expected 500 records in dataset, found {len(df)}"
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected null values."


# Phase 5: Database Operations

def test_database_initialization_and_seeding():
    with get_db_connection() as conn:
        users_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        garages_count = conn.execute("SELECT COUNT(*) FROM garages").fetchone()[0]
        parts_count = conn.execute("SELECT COUNT(*) FROM spare_parts").fetchone()[0]
        requests_count = conn.execute("SELECT COUNT(*) FROM service_requests").fetchone()[0]

    assert users_count >= 2, "Default demo accounts not seeded."
    assert garages_count >= 7, "Default garages not seeded."
    assert parts_count == 500, "Spare parts from CSV not seeded into SQLite."
    assert requests_count >= 3, "Demo service requests not seeded."


def test_database_spare_parts_filtering():
    results = search_spare_parts(part_name="Clutch Cable")
    assert len(results) > 0, "Should find records for Clutch Cable."
    for r in results:
        assert "clutch" in r["part_name"].lower()


# Phase 10: Health & Core Endpoints

def test_health_endpoints(client):
    res1 = client.get("/health")
    assert res1.status_code == 200
    assert res1.get_json()["status"] == "ok"

    res2 = client.get("/api/health")
    assert res2.status_code == 200
    assert res2.get_json()["status"] == "ok"


def test_home_page(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Smart Garage" in html
    assert "Thunderbird 350X" in html


# Phase 6: Authentication & Two Login Options

def test_login_selection_page(client):
    res = client.get("/login")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "USER LOGIN" in html
    assert "GARAGE LOGIN" in html


def test_user_login_success(client):
    res = client.post(
        "/login/user",
        data={"username": "user_demo", "password": "password123"},
        follow_redirects=False,
    )
    assert res.status_code == 302
    assert "/user/dashboard" in res.headers["Location"] or "/dashboard" in res.headers["Location"]


def test_user_login_invalid_credentials(client):
    res = client.post(
        "/login/user",
        data={"username": "user_demo", "password": "wrongpassword"},
        follow_redirects=True,
    )
    assert res.status_code == 200
    assert "Invalid username or password" in res.get_data(as_text=True)


def test_garage_login_success(client):
    res = client.post(
        "/login/garage",
        data={"username": "garage_demo", "password": "garage123"},
        follow_redirects=False,
    )
    assert res.status_code == 302
    assert "/garage/dashboard" in res.headers["Location"] or "/dashboard" in res.headers["Location"]


def test_role_based_access_protection(client):
    # Unauthenticated user access to protected routes
    res = client.get("/user/dashboard", follow_redirects=False)
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]

    res_garage = client.get("/garage/dashboard", follow_redirects=False)
    assert res_garage.status_code == 302
    assert "/login" in res_garage.headers["Location"]

    # Log in as rider/user
    client.post("/login/user", data={"username": "user_demo", "password": "password123"})
    # Rider should NOT access garage dashboard
    res_forbidden = client.get("/garage/dashboard", follow_redirects=False)
    assert res_forbidden.status_code == 302
    assert "/user/dashboard" in res_forbidden.headers["Location"]


# Phase 4: Machine Learning Inference & Storage

def test_ml_model_files_present():
    expected = [
        PROJECT_ROOT / "ML" / "models" / "availability_model.pkl",
        PROJECT_ROOT / "ML" / "models" / "wait_model.pkl",
        PROJECT_ROOT / "ML" / "models" / "price_model.pkl",
        PROJECT_ROOT / "ML" / "models" / "metrics.json",
    ]
    for p in expected:
        assert p.exists(), f"Model artifact {p.name} does not exist."


def test_availability_prediction_api(client):
    payload = {
        "bike_model": "Royal Enfield Thunderbird 350X",
        "manufacturing_year": 2019,
        "part_name": "Clutch Cable",
        "part_category": "Control",
        "supplier_type": "Online Supplier",
        "city": "Pune",
        "garage_type": "Royal Enfield Specialist",
        "part_condition": "New",
        "service_priority": "High",
    }
    res = client.post("/api/predict-availability", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "availability" in data
    assert data["availability"] in ["Available", "Limited", "Discontinued"]
    assert "confidence" in data


def test_wait_and_price_prediction_api(client):
    payload = {
        "bike_model": "Royal Enfield Thunderbird 350X",
        "manufacturing_year": 2019,
        "part_name": "Front Disc Brake Pads",
        "part_category": "Braking",
        "supplier_type": "Authorized Dealer",
        "city": "Pune",
        "garage_type": "Royal Enfield Specialist",
        "part_condition": "New",
        "service_priority": "Medium",
    }
    res_wait = client.post("/api/predict-wait", json=payload)
    assert res_wait.status_code == 200
    assert "estimated_wait_days" in res_wait.get_json()
    assert res_wait.get_json()["estimated_wait_days"] >= 0

    res_price = client.post("/api/predict-price", json=payload)
    assert res_price.status_code == 200
    assert "estimated_price_inr" in res_price.get_json()
    assert res_price.get_json()["estimated_price_inr"] > 0


def test_predict_part_details_combined(client):
    payload = {
        "part_name": "Ignition Coil",
        "part_category": "Electrical",
        "city": "Pune",
        "manufacturing_year": 2019,
    }
    res = client.post("/api/predict-part-details", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "availability" in data
    assert "estimated_wait_days" in data
    assert "estimated_price_inr" in data


def test_invalid_prediction_payloads(client):
    res_empty = client.post("/api/predict-availability", json={})
    assert res_empty.status_code == 400

    res_year = client.post("/api/predict-availability", json={"manufacturing_year": "invalid_year"})
    assert res_year.status_code == 400


# Phase 7 & 8: Search and Garage Recommendations

def test_parts_search_view(client):
    res = client.get("/parts?part_name=Brake")
    assert res.status_code == 200
    assert "Brake" in res.get_data(as_text=True)


def test_find_garages_api(client):
    res = client.post("/api/find-garages", json={"city": "Pune", "part_name": "Clutch Cable"})
    assert res.status_code == 200
    data = res.get_json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0
    assert "recommendation_score" in data["recommendations"][0]


# Phase 9: Service Request Workflow

def test_service_request_workflow(client):
    # 1. User submits service request via API
    payload = {
        "name": "Rider Rohan",
        "mobile": "+91 98888 77777",
        "bike_model": "Royal Enfield Thunderbird 350X",
        "part_required": "Throttle Cable",
        "preferred_garage": "Bullet Masters Pune",
        "service_priority": "High",
        "city": "Pune",
        "request_description": "Sticky throttle response.",
    }
    res = client.post("/api/service-request", json=payload)
    assert res.status_code == 201
    req_data = res.get_json()["request"]
    req_id = req_data["request_id"]
    assert req_id.startswith("SR-")

    # 2. Check request in database
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM service_requests WHERE request_id = ?", (req_id,)).fetchone()
        assert row is not None
        assert row["status"] == "Pending"

    # 3. Garage updates status to 'Accepted' and adds mechanic notes
    updated = update_service_request_status(
        request_id=req_id,
        garage_id=1,
        new_status="Accepted",
        notes="Throttle cable in stock. Scheduled for 11 AM.",
    )
    assert updated is True

    # 4. Verify status updated in DB
    with get_db_connection() as conn:
        updated_row = conn.execute("SELECT * FROM service_requests WHERE request_id = ?", (req_id,)).fetchone()
        assert updated_row["status"] == "Accepted"
        assert "Throttle cable in stock" in updated_row["garage_notes"]


def test_unauthorized_garage_status_update():
    # If a request is already assigned to Garage 1, Garage 2 cannot update it
    req_id = create_service_request(
        user_id=1,
        user_name="Test Rider",
        user_phone="9999999999",
        bike_model="Royal Enfield Thunderbird 350X",
        part_name="Battery 12V 14Ah",
        city="Pune",
        priority="Medium",
        garage_id=1,
    )
    # Attempt update by different garage_id (e.g. 2)
    result = update_service_request_status(req_id, garage_id=2, new_status="Completed")
    assert result is False, "Garage 2 should not be able to update a request assigned to Garage 1."


def test_ml_metrics_page_and_api(client):
    res_page = client.get("/ml-metrics")
    assert res_page.status_code == 200
    assert "Evaluation" in res_page.get_data(as_text=True)

    res_api = client.get("/api/ml-metrics")
    assert res_api.status_code == 200
    assert "classification" in res_api.get_json()
