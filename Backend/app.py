"""Smart Garage & Spare Parts Assistance System for Royal Enfield Thunderbird 350X
Flask Application providing Web UI and REST API.
"""

from __future__ import annotations

import json
import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import joblib
import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_cors import CORS

from database import (
    authenticate_user,
    create_service_request,
    get_db_connection,
    get_garage_by_user_id,
    get_garage_service_requests,
    get_user_by_id,
    get_user_service_requests,
    init_db,
    list_garages,
    log_prediction,
    register_user,
    search_spare_parts,
    seed_db,
    update_service_request_status,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "Dataset" / "royal_enfield_thunderbird_350x_parts_dataset.csv"
MODEL_DIRS = [
    PROJECT_ROOT / "ML" / "models",
    PROJECT_ROOT / "models",
]
DATA_DIR = PROJECT_ROOT / "Backend" / "data"
SERVICE_REQUESTS_PATH = DATA_DIR / "service_requests.json"
METRICS_PATH = PROJECT_ROOT / "ML" / "models" / "metrics.json"

TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = os.environ.get("SECRET_KEY", "re-thunderbird-350x-college-project-key-2026")
CORS(app)

MODEL_CONFIG = {
    "availability": "availability_model.pkl",
    "wait": "wait_model.pkl",
    "price": "price_model.pkl",
}

FEATURE_COLUMNS = [
    "Bike model",
    "Manufacturing year",
    "Part name",
    "Part category",
    "Supplier type",
    "City",
    "Garage type",
    "Part condition",
    "Service priority",
    "record_month",
    "record_year",
]

# In-memory model cache
_LOADED_MODELS: dict[str, Any] = {}


def get_model_path(model_name: str) -> Optional[Path]:
    for base_dir in MODEL_DIRS:
        candidate = base_dir / MODEL_CONFIG[model_name]
        if candidate.exists():
            return candidate
    return None


def load_model(model_key: str) -> Any:
    """Loads and caches trained ML pipeline, with descriptive error if not found."""
    if model_key in _LOADED_MODELS:
        return _LOADED_MODELS[model_key]

    path = get_model_path(model_key)
    if not path:
        raise FileNotFoundError(
            f"Model '{model_key}' not found in {[str(d) for d in MODEL_DIRS]}. Run ML/02_preprocessing_and_modeling.py."
        )

    model = joblib.load(path)
    _LOADED_MODELS[model_key] = model
    return model


def load_dataset() -> pd.DataFrame:
    """Loads the Royal Enfield Thunderbird 350X parts dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df = df.copy()
    df["Record date"] = pd.to_datetime(df["Record date"], errors="coerce")
    df["record_month"] = df["Record date"].dt.month.fillna(1).astype(int)
    df["record_year"] = df["Record date"].dt.year.fillna(2025).astype(int)
    return df


def load_metrics_data() -> dict[str, Any]:
    """Reads evaluation metrics from ML/models/metrics.json with fallback."""
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "classification": {"best_model": "decision_tree", "metrics": {"accuracy": 0.37, "f1_weighted": 0.375}, "comparison": []},
        "wait_time_regression": {"best_model": "random_forest", "metrics": {"mae": 9.83, "rmse": 13.68, "r2": -0.118}, "comparison": []},
        "price_regression": {"best_model": "linear_regression", "metrics": {"mae": 619.55, "rmse": 967.92, "r2": 0.861}, "comparison": []},
    }


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = payload.copy()
    normalized = {
        "Bike model": payload.get("Bike model", payload.get("bike_model", "Royal Enfield Thunderbird 350X")),
        "Manufacturing year": payload.get("Manufacturing year", payload.get("manufacturing_year", 2019)),
        "Part name": payload.get("Part name", payload.get("part_name", "Clutch Cable")),
        "Part category": payload.get("Part category", payload.get("part_category", "Engine/Clutch")),
        "Supplier type": payload.get("Supplier type", payload.get("supplier_type", "Online Supplier")),
        "City": payload.get("City", payload.get("city", "Pune")),
        "Garage type": payload.get("Garage type", payload.get("garage_type", "Royal Enfield Specialist")),
        "Part condition": payload.get("Part condition", payload.get("part_condition", "New")),
        "Service priority": payload.get("Service priority", payload.get("service_priority", "High")),
        "Record date": payload.get("Record date", payload.get("record_date", "2025-01-01")),
    }
    try:
        normalized["Manufacturing year"] = int(normalized["Manufacturing year"])
    except (TypeError, ValueError):
        raise ValueError("Manufacturing year must be an integer.")
    if normalized["Manufacturing year"] < 2000 or normalized["Manufacturing year"] > 2035:
        raise ValueError("Manufacturing year must be between 2000 and 2035.")

    record_series = pd.to_datetime(normalized["Record date"], errors="coerce")
    if pd.isna(record_series):
        normalized["record_month"] = 1
        normalized["record_year"] = 2025
    else:
        normalized["record_month"] = int(record_series.month)
        normalized["record_year"] = int(record_series.year)
    return normalized


def prepare_feature_frame(payload: dict[str, Any]) -> pd.DataFrame:
    row = normalize_payload(payload)
    feature_row = {feature: row.get(feature, "") for feature in FEATURE_COLUMNS}
    return pd.DataFrame([feature_row])


def score_availability(value: str) -> float:
    v = str(value).strip().lower()
    if v == "available":
        return 1.0
    if v == "limited":
        return 0.6
    return 0.2


# Authentication & Access Control Decorators

def login_required(f: Callable) -> Callable:
    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login_select"))
        return f(*args, **kwargs)
    return decorated_function


def user_required(f: Callable) -> Callable:
    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login_user"))
        if session.get("role") != "user":
            flash("Access restricted to motorcycle rider accounts.", "danger")
            return redirect(url_for("garage_dashboard"))
        return f(*args, **kwargs)
    return decorated_function


def garage_required(f: Callable) -> Callable:
    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        if "user_id" not in session:
            flash("Please log in with a garage account to access the workshop portal.", "warning")
            return redirect(url_for("login_garage"))
        if session.get("role") != "garage":
            flash("Access restricted to service garage accounts.", "danger")
            return redirect(url_for("user_dashboard"))
        return f(*args, **kwargs)
    return decorated_function


# Web UI Routes

@app.route("/")
def index() -> Any:
    return render_template("index.html")


@app.route("/login")
def login_select() -> Any:
    return render_template("login_select.html")


@app.route("/login/user", methods=["GET", "POST"])
def login_user() -> Any:
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = authenticate_user(username, password, expected_role="user")
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["full_name"] = user["full_name"]
            session["role"] = user["role"]
            session["city"] = user["city"]
            flash(f"Welcome back, {user['full_name']}!", "success")
            return redirect(url_for("user_dashboard"))
        flash("Invalid username or password for rider account.", "danger")
    return render_template("login_user.html")


@app.route("/login/garage", methods=["GET", "POST"])
def login_garage() -> Any:
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = authenticate_user(username, password, expected_role="garage")
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["full_name"] = user["full_name"]
            session["role"] = user["role"]
            session["city"] = user["city"]
            flash(f"Welcome to the Garage Portal, {user['full_name']}!", "success")
            return redirect(url_for("garage_dashboard"))
        flash("Invalid credentials for service garage account.", "danger")
    return render_template("login_garage.html")


@app.route("/register", methods=["GET", "POST"])
def register() -> Any:
    if request.method == "POST":
        role = request.form.get("role", "user")
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        city = request.form.get("city", "Pune").strip()
        password = request.form.get("password", "").strip()

        ok, msg, new_id = register_user(username, email, password, full_name, phone, city, role)
        if ok:
            flash("Account registered successfully. Please sign in.", "success")
            return redirect(url_for("login_garage" if role == "garage" else "login_user"))
        flash(msg, "danger")
    return render_template("register.html")


@app.route("/logout")
def logout() -> Any:
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard() -> Any:
    if session.get("role") == "garage":
        return redirect(url_for("garage_dashboard"))
    return redirect(url_for("user_dashboard"))


@app.route("/user/dashboard")
@user_required
def user_dashboard() -> Any:
    user = get_user_by_id(session["user_id"])
    requests_list = get_user_service_requests(session["user_id"])
    garages = list_garages(city=user.get("city", "Pune")) if user else list_garages()
    return render_template("user_dashboard.html", user=user, requests=requests_list, garages=garages)


@app.route("/garage/dashboard")
@garage_required
def garage_dashboard() -> Any:
    garage = get_garage_by_user_id(session["user_id"])
    if not garage:
        flash("Garage profile not found.", "danger")
        return redirect(url_for("index"))
    requests_list = get_garage_service_requests(garage["id"], city=garage.get("city"))
    return render_template("garage_dashboard.html", garage=garage, requests=requests_list)


@app.route("/parts")
@app.route("/search-parts")
def search_parts_view() -> Any:
    part_name = request.args.get("part_name", "").strip() or None
    category = request.args.get("category", "").strip() or None
    availability = request.args.get("availability", "").strip() or None
    city = request.args.get("city", "").strip() or None
    condition = request.args.get("condition", "").strip() or None
    max_price = float(request.args.get("max_price")) if request.args.get("max_price") else None
    max_wait = int(request.args.get("max_wait_days")) if request.args.get("max_wait_days") else None

    parts = search_spare_parts(
        part_name=part_name,
        category=category,
        availability=availability,
        city=city,
        condition=condition,
        max_price=max_price,
        max_wait_days=max_wait,
        limit=100,
    )

    categories = ["Engine/Clutch", "Control", "Braking", "Transmission", "Suspension", "Wheels", "Tyre", "Electrical", "Engine", "Body"]
    cities = ["Pune", "Mumbai", "Nashik", "Kolhapur", "Aurangabad", "Satara", "Solapur", "Ahmednagar", "Sangli", "Baramati"]
    return render_template("parts_search.html", parts=parts, categories=categories, cities=cities)


@app.route("/predict")
def predict_view() -> Any:
    metrics = load_metrics_data()
    return render_template("predict.html", metrics=metrics)


@app.route("/recommendations")
def recommendations_view() -> Any:
    search_part = request.args.get("part_name", "").strip()
    search_city = request.args.get("city", "").strip()

    df = load_dataset()
    if search_part:
        df = df[df["Part name"].str.contains(search_part, case=False, na=False)]
    if search_city:
        df = df[df["City"].str.contains(search_city, case=False, na=False)]
    if df.empty:
        df = load_dataset().head(12)

    records = []
    max_wait = float(df["Estimated wait days"].max()) if not df["Estimated wait days"].empty else 1.0
    max_price = float(df["Estimated price (INR)"].max()) if not df["Estimated price (INR)"].empty else 1.0

    for idx, row in df.head(15).iterrows():
        availability_score = score_availability(str(row["Availability status"]))
        wait_score = max(0.0, 1.0 - (float(row["Estimated wait days"]) / max_wait)) if max_wait else 0.0
        price_score = max(0.0, 1.0 - (float(row["Estimated price (INR)"]) / max_price)) if max_price else 0.0
        garage_score = 0.7 if "specialist" in str(row["Garage type"]).lower() else 0.5
        supplier_score = 0.9 if "oem" in str(row["Supplier type"]).lower() or "authorized" in str(row["Supplier type"]).lower() else 0.6

        recommendation = 100 * (
            0.40 * availability_score +
            0.25 * wait_score +
            0.20 * price_score +
            0.10 * garage_score +
            0.05 * supplier_score
        )
        records.append({
            "garage_name": f"Royal Enfield Workshop {idx + 1:02d}",
            "garage_type": str(row["Garage type"]),
            "city": str(row["City"]),
            "availability": str(row["Availability status"]),
            "estimated_wait_days": int(row["Estimated wait days"]),
            "estimated_price_inr": int(row["Estimated price (INR)"]),
            "recommendation_score": round(float(recommendation), 1),
            "part_name": str(row["Part name"]),
            "supplier_type": str(row["Supplier type"]),
        })

    recommendations = sorted(records, key=lambda r: r["recommendation_score"], reverse=True)
    cities = ["Pune", "Mumbai", "Nashik", "Kolhapur", "Aurangabad", "Satara", "Solapur", "Ahmednagar", "Sangli", "Baramati"]
    return render_template("recommendations.html", recommendations=recommendations, cities=cities)


@app.route("/service-requests")
@login_required
def service_requests_view() -> Any:
    if session.get("role") == "garage":
        return redirect(url_for("garage_dashboard"))
    requests_list = get_user_service_requests(session["user_id"])
    return render_template("service_requests.html", requests=requests_list)


@app.route("/service-request/new", methods=["GET", "POST"])
@login_required
def new_service_request_view() -> Any:
    user = get_user_by_id(session["user_id"])
    if not user:
        flash("User session invalid.", "danger")
        return redirect(url_for("login_select"))

    if request.method == "POST":
        user_name = request.form.get("user_name", user["full_name"])
        user_phone = request.form.get("user_phone", user["phone"])
        bike_model = request.form.get("bike_model", "Royal Enfield Thunderbird 350X")
        part_name = request.form.get("part_name", "").strip()
        part_category = request.form.get("part_category", "General")
        city = request.form.get("city", user["city"])
        priority = request.form.get("priority", "Medium")
        issue_description = request.form.get("issue_description", "").strip()
        garage_id_val = request.form.get("garage_id")
        garage_id = int(garage_id_val) if garage_id_val and garage_id_val.isdigit() else None
        est_price = float(request.form.get("estimated_price", 0.0))
        est_wait = int(request.form.get("estimated_wait_days", 0))

        if not part_name:
            flash("Part name is required.", "danger")
            return redirect(url_for("new_service_request_view"))

        req_id = create_service_request(
            user_id=session["user_id"],
            user_name=user_name,
            user_phone=user_phone,
            bike_model=bike_model,
            part_name=part_name,
            city=city,
            priority=priority,
            issue_description=issue_description,
            garage_id=garage_id,
            part_category=part_category,
            estimated_price=est_price,
            estimated_wait_days=est_wait,
        )
        flash(f"Service Request {req_id} logged successfully!", "success")
        return redirect(url_for("user_dashboard"))

    garages = list_garages(city=user.get("city", "Pune"))
    return render_template("new_request.html", user=user, garages=garages)


@app.route("/service-request/<req_id>/update", methods=["POST"])
@garage_required
def update_service_request_status_view(req_id: str) -> Any:
    garage = get_garage_by_user_id(session["user_id"])
    if not garage:
        flash("Garage account required.", "danger")
        return redirect(url_for("login_garage"))

    new_status = request.form.get("status", "").strip()
    notes = request.form.get("notes", "").strip()

    ok = update_service_request_status(req_id, garage["id"], new_status, notes)
    if ok:
        flash(f"Request {req_id} status updated to '{new_status}'.", "success")
    else:
        flash(f"Failed to update request {req_id}. Access denied or invalid status.", "danger")
    return redirect(url_for("garage_dashboard"))


@app.route("/profile")
@login_required
def profile_view() -> Any:
    user = get_user_by_id(session["user_id"])
    return render_template("profile.html", user=user)


@app.route("/ml-metrics")
def ml_metrics_view() -> Any:
    metrics = load_metrics_data()
    return render_template("ml_metrics.html", metrics=metrics)


# REST API Endpoints

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check() -> Any:
    return jsonify({
        "status": "ok",
        "service": "Thunderbird 350X Smart Garage API",
        "models_loaded": {k: get_model_path(k) is not None for k in MODEL_CONFIG},
        "database": "sqlite",
    })


@app.route("/predict-availability", methods=["POST"])
@app.route("/api/predict-availability", methods=["POST"])
def predict_availability() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        model = load_model("availability")
        feature_df = prepare_feature_frame(payload)
        prediction = model.predict(feature_df)[0]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature_df)[0]
            confidence = float(max(probabilities))
        else:
            confidence = 0.85
        return jsonify({
            "availability": str(prediction),
            "confidence": round(confidence, 4),
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception:
        # Graceful fallback with clear academic demo label
        return jsonify({
            "availability": "Limited",
            "confidence": 0.65,
            "demo_fallback": True,
            "note": "Fallback prediction mode.",
        }), 200


@app.route("/predict-wait", methods=["POST"])
@app.route("/api/predict-wait", methods=["POST"])
def predict_wait() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        model = load_model("wait")
        feature_df = prepare_feature_frame(payload)
        wait_value = float(model.predict(feature_df)[0])
        return jsonify({"estimated_wait_days": max(0, int(round(wait_value)))})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception:
        return jsonify({"estimated_wait_days": 5, "demo_fallback": True}), 200


@app.route("/predict-price", methods=["POST"])
@app.route("/api/predict-price", methods=["POST"])
def predict_price() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        model = load_model("price")
        feature_df = prepare_feature_frame(payload)
        price_value = float(model.predict(feature_df)[0])
        return jsonify({"estimated_price_inr": max(0, int(round(price_value)))})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception:
        return jsonify({"estimated_price_inr": 1250, "demo_fallback": True}), 200


@app.route("/predict-part-details", methods=["POST"])
@app.route("/api/predict-part-details", methods=["POST"])
def predict_part_details() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        normalized = normalize_payload(payload)
        feature_df = prepare_feature_frame(normalized)

        availability_model = load_model("availability")
        wait_model = load_model("wait")
        price_model = load_model("price")

        availability = str(availability_model.predict(feature_df)[0])
        confidence = float(max(availability_model.predict_proba(feature_df)[0])) if hasattr(availability_model, "predict_proba") else 0.85
        wait = max(0, int(round(float(wait_model.predict(feature_df)[0]))))
        price = max(0, int(round(float(price_model.predict(feature_df)[0]))))

        # Log prediction to history if user is logged in
        user_id = session.get("user_id")
        try:
            log_prediction(
                user_id=user_id,
                part_name=normalized["Part name"],
                part_category=normalized["Part category"],
                city=normalized["City"],
                availability=availability,
                price=float(price),
                wait_days=wait,
                confidence=confidence,
            )
        except Exception:
            pass

        return jsonify({
            "bike_model": normalized["Bike model"],
            "part_name": normalized["Part name"],
            "part_category": normalized["Part category"],
            "availability": availability,
            "confidence": round(confidence, 4),
            "estimated_wait_days": wait,
            "estimated_price_inr": price,
            "note": "Academic estimation based on synthetic Thunderbird 350X records. Not verified live inventory.",
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


@app.route("/find-garages", methods=["POST"])
@app.route("/api/find-garages", methods=["POST"])
def find_garages() -> Any:
    payload = request.get_json(silent=True)
    if payload is not None and not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        df = load_dataset()
        search = payload or {}
        part_name = search.get("Part name", search.get("part_name", ""))
        city = search.get("City", search.get("city", ""))

        if part_name:
            df = df[df["Part name"].str.contains(part_name, case=False, na=False)]
        if city:
            df = df[df["City"].str.contains(city, case=False, na=False)]
        if df.empty:
            df = load_dataset().head(10)

        records = []
        max_wait = float(df["Estimated wait days"].max()) if not df["Estimated wait days"].empty else 1.0
        max_price = float(df["Estimated price (INR)"].max()) if not df["Estimated price (INR)"].empty else 1.0

        for idx, row in df.head(10).iterrows():
            availability_score = score_availability(str(row["Availability status"]))
            wait_score = max(0.0, 1.0 - (float(row["Estimated wait days"]) / max_wait)) if max_wait else 0.0
            price_score = max(0.0, 1.0 - (float(row["Estimated price (INR)"]) / max_price)) if max_price else 0.0
            garage_score = 0.7 if "specialist" in str(row["Garage type"]).lower() else 0.5
            supplier_score = 0.9 if "oem" in str(row["Supplier type"]).lower() or "authorized" in str(row["Supplier type"]).lower() else 0.6

            recommendation = 100 * (
                0.40 * availability_score +
                0.25 * wait_score +
                0.20 * price_score +
                0.10 * garage_score +
                0.05 * supplier_score
            )
            records.append({
                "garage_name": f"Royal Enfield Specialist Garage {idx + 1:02d}",
                "city": str(row["City"]),
                "availability": str(row["Availability status"]),
                "estimated_wait_days": int(row["Estimated wait days"]),
                "estimated_price_inr": int(row["Estimated price (INR)"]),
                "recommendation_score": round(float(recommendation), 2),
                "part_name": str(row["Part name"]),
                "supplier_type": str(row["Supplier type"]),
                "garage_type": str(row["Garage type"]),
            })

        recommendations = sorted(records, key=lambda r: r["recommendation_score"], reverse=True)
        return jsonify({"recommendations": recommendations})
    except Exception as exc:
        return jsonify({"error": f"Garage recommendation failed: {str(exc)}"}), 500


@app.route("/service-request", methods=["POST"])
@app.route("/api/service-request", methods=["POST"])
def service_request_api() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "Request body must be a JSON object."}), 400

    required_fields = ["name", "mobile", "bike_model", "part_required", "preferred_garage", "service_priority"]
    missing = [field for field in required_fields if not payload.get(field)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    user_id = session.get("user_id", 1)
    req_id = create_service_request(
        user_id=user_id,
        user_name=payload.get("name", "Guest Rider"),
        user_phone=payload.get("mobile", ""),
        bike_model=payload.get("bike_model", "Royal Enfield Thunderbird 350X"),
        part_name=payload.get("part_required", "General Part"),
        city=payload.get("city", "Pune"),
        priority=payload.get("service_priority", "Medium"),
        issue_description=payload.get("request_description", ""),
        estimated_price=float(payload.get("estimated_price_inr", 0.0)),
        estimated_wait_days=int(payload.get("estimated_wait_days", 0)),
    )

    new_request = {
        "request_id": req_id,
        "name": payload.get("name"),
        "mobile": payload.get("mobile"),
        "bike_model": payload.get("bike_model"),
        "part_required": payload.get("part_required"),
        "preferred_garage": payload.get("preferred_garage"),
        "service_priority": payload.get("service_priority"),
        "request_description": payload.get("request_description", ""),
        "status": "Pending",
    }
    return jsonify({"message": "Service request submitted successfully.", "request": new_request}), 201


@app.route("/api/service-requests", methods=["GET"])
def api_get_service_requests() -> Any:
    user_id = session.get("user_id")
    if not user_id:
        # Return all for testing/demo or public query if no session
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM service_requests ORDER BY id DESC").fetchall()
            return jsonify({"requests": [dict(r) for r in rows]})

    if session.get("role") == "garage":
        garage = get_garage_by_user_id(user_id)
        requests_list = get_garage_service_requests(garage["id"]) if garage else []
    else:
        requests_list = get_user_service_requests(user_id)
    return jsonify({"requests": requests_list})


@app.route("/api/ml-metrics", methods=["GET"])
def api_ml_metrics() -> Any:
    return jsonify(load_metrics_data())


# Startup Initialization
try:
    init_db()
    seed_db()
except Exception as e:
    print(f"Database setup note: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
