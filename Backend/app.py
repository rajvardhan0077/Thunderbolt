from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "Dataset" / "royal_enfield_thunderbird_350x_parts_dataset.csv"
MODEL_DIRS = [
    PROJECT_ROOT / "ML" / "models",
    PROJECT_ROOT / "models",
]
DATA_DIR = PROJECT_ROOT / "Backend" / "data"
SERVICE_REQUESTS_PATH = DATA_DIR / "service_requests.json"

app = Flask(__name__)
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


def ensure_data_store() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SERVICE_REQUESTS_PATH.exists():
        SERVICE_REQUESTS_PATH.write_text("[]", encoding="utf-8")


def get_model_path(model_name: str) -> Path:
    for base_dir in MODEL_DIRS:
        candidate = base_dir / MODEL_CONFIG[model_name]
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Model '{model_name}' not found. Run ML/02_preprocessing_and_modeling.py first."
    )


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df = df.copy()
    df["Record date"] = pd.to_datetime(df["Record date"], errors="coerce")
    df["record_month"] = df["Record date"].dt.month.fillna(1)
    df["record_year"] = df["Record date"].dt.year.fillna(2025)
    return df


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
    if value == "Available":
        return 1.0
    if value == "Limited":
        return 0.6
    return 0.2


def load_model(model_key: str):
    return joblib.load(get_model_path(model_key))


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/predict-availability", methods=["POST"])
def predict_availability():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        model = load_model("availability")
        feature_df = prepare_feature_frame(payload)
        prediction = model.predict(feature_df)[0]
        probabilities = model.predict_proba(feature_df)[0]
        confidence = float(max(probabilities))
        return jsonify({
            "availability": str(prediction),
            "confidence": round(confidence, 4),
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": "Prediction failed. Please validate your input."}), 500


@app.route("/predict-wait", methods=["POST"])
def predict_wait():
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
        return jsonify({"error": "Waiting time prediction failed."}), 500


@app.route("/predict-price", methods=["POST"])
def predict_price():
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
        return jsonify({"error": "Price estimation failed."}), 500


@app.route("/predict-part-details", methods=["POST"])
def predict_part_details():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        normalized = normalize_payload(payload)
        availability = load_model("availability").predict(prepare_feature_frame(normalized))[0]
        wait = load_model("wait").predict(prepare_feature_frame(normalized))[0]
        price = load_model("price").predict(prepare_feature_frame(normalized))[0]
        return jsonify({
            "bike_model": normalized["Bike model"],
            "part_name": normalized["Part name"],
            "part_category": normalized["Part category"],
            "availability": str(availability),
            "estimated_wait_days": max(0, int(round(float(wait)))),
            "estimated_price_inr": max(0, int(round(float(price)))),
            "note": "Prototype estimate based on the demo dataset. Not live inventory.",
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception:
        return jsonify({"error": "Part detail prediction failed."}), 500


@app.route("/find-garages", methods=["POST"])
def find_garages():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        df = load_dataset()
        search = payload or {}
        if "part_name" in search or "Part name" in search:
            part_name = search.get("Part name", search.get("part_name", ""))
            df = df[df["Part name"].str.contains(part_name, case=False, na=False)]
        if "city" in search or "City" in search:
            city = search.get("City", search.get("city", ""))
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
            garage_score = 0.7 if str(row["Garage type"]).lower().find("specialist") >= 0 else 0.5
            supplier_score = 0.9 if str(row["Supplier type"]).lower().find("oem") >= 0 else 0.6
            recommendation = (
                100
                * (0.4 * availability_score + 0.25 * wait_score + 0.2 * price_score + 0.1 * garage_score + 0.05 * supplier_score)
            )
            records.append({
                "garage_name": f"Garage {idx + 1:03d}",
                "city": str(row["City"]),
                "availability": str(row["Availability status"]),
                "estimated_wait_days": int(row["Estimated wait days"]),
                "estimated_price_inr": int(row["Estimated price (INR)"]),
                "recommendation_score": round(float(recommendation), 2),
                "part_name": str(row["Part name"]),
                "supplier_type": str(row["Supplier type"]),
            })

        recommendations = sorted(records, key=lambda r: r["recommendation_score"], reverse=True)
        return jsonify({"recommendations": recommendations})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception:
        return jsonify({"error": "Garage recommendation failed."}), 500


@app.route("/service-request", methods=["POST"])
def service_request():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "Request body must be a JSON object."}), 400
    required_fields = ["name", "mobile", "bike_model", "part_required", "preferred_garage", "service_priority"]
    missing = [field for field in required_fields if not payload.get(field)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    ensure_data_store()
    try:
        requests_list = json.loads(SERVICE_REQUESTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        requests_list = []

    request_id = f"SR-{len(requests_list) + 1:03d}"
    new_request = {
        "request_id": request_id,
        "name": payload.get("name"),
        "mobile": payload.get("mobile"),
        "bike_model": payload.get("bike_model"),
        "part_required": payload.get("part_required"),
        "preferred_garage": payload.get("preferred_garage"),
        "service_priority": payload.get("service_priority"),
        "request_description": payload.get("request_description", ""),
        "status": "submitted",
    }
    requests_list.append(new_request)
    SERVICE_REQUESTS_PATH.write_text(json.dumps(requests_list, indent=2), encoding="utf-8")
    return jsonify({"message": "Service request submitted successfully.", "request": new_request}), 201


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
