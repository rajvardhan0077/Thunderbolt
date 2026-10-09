from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "Dataset" / "royal_enfield_thunderbird_350x_parts_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "ML" / "models"
LEGACY_MODEL_DIR = PROJECT_ROOT / "models"
PROJECT_MODEL_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "Availability status"
AVAILABILITY_LABELS = ["Available", "Limited", "Discontinued"]

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
NUMERIC_COLUMNS = [
    "Manufacturing year",
    "record_month",
    "record_year",
]
WAIT_TARGET = "Estimated wait days"
PRICE_TARGET = "Estimated price (INR)"


def ensure_model_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LEGACY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    required_columns = {
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
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    df = df.copy()
    df["Record date"] = pd.to_datetime(df["Record date"], errors="coerce")
    df["record_month"] = df["Record date"].dt.month.fillna(1)
    df["record_year"] = df["Record date"].dt.year.fillna(2025)
    df["Availability status"] = df["Availability status"].astype(str).str.strip()
    return df


def build_preprocessor(feature_columns: list[str] | None = None) -> ColumnTransformer:
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS.copy()

    categorical_features = [col for col in feature_columns if col not in NUMERIC_COLUMNS]
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, [col for col in feature_columns if col in NUMERIC_COLUMNS]),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def build_classification_model(name: str) -> Pipeline:
    preprocessor = build_preprocessor(FEATURE_COLUMNS)
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2500, multi_class="auto")
    elif name == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=42)
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
        )
    else:
        raise ValueError(f"Unsupported availability model: {name}")
    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def build_regression_model(name: str, target: str) -> Pipeline:
    preprocessor = build_preprocessor(FEATURE_COLUMNS)
    if name == "linear_regression":
        estimator = LinearRegression()
    elif name == "decision_tree":
        estimator = DecisionTreeRegressor(random_state=42)
    elif name == "random_forest":
        estimator = RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        raise ValueError(f"Unsupported regression model: {name} for {target}")
    return Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])


def evaluate_classification_model(model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    model = build_classification_model(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "model": model_name,
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "classification_report": report,
    }


def evaluate_regression_model(model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    model = build_regression_model(model_name, "target")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(mse ** 0.5)
    r2 = r2_score(y_test, y_pred)
    return {
        "model": model_name,
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }


def train_availability_model(df: pd.DataFrame) -> dict[str, Any]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_names = ["logistic_regression", "decision_tree", "random_forest"]
    results = [
        evaluate_classification_model(name, X_train, X_test, y_train, y_test)
        for name in model_names
    ]
    best = max(results, key=lambda item: (item["f1_weighted"], item["accuracy"]))
    best_model_name = best["model"]
    best_model = build_classification_model(best_model_name)
    best_model.fit(X_train, y_train)

    ensure_model_dirs()
    joblib.dump(best_model, MODEL_DIR / "availability_model.pkl")
    joblib.dump(best_model, LEGACY_MODEL_DIR / "availability_model.pkl")
    joblib.dump(best_model, PROJECT_MODEL_DIR / "availability_model.pkl")
    return {
        "best_model": best_model_name,
        "comparison": results,
        "metrics": best,
    }


def train_wait_model(df: pd.DataFrame) -> dict[str, Any]:
    X = df[FEATURE_COLUMNS]
    y = df[WAIT_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model_names = ["linear_regression", "decision_tree", "random_forest"]
    results = [
        evaluate_regression_model(name, X_train, X_test, y_train, y_test)
        for name in model_names
    ]
    best = max(results, key=lambda item: (item["r2"], -item["rmse"]))
    best_model_name = best["model"]
    best_model = build_regression_model(best_model_name, WAIT_TARGET)
    best_model.fit(X_train, y_train)

    ensure_model_dirs()
    joblib.dump(best_model, MODEL_DIR / "wait_model.pkl")
    joblib.dump(best_model, LEGACY_MODEL_DIR / "wait_model.pkl")
    joblib.dump(best_model, PROJECT_MODEL_DIR / "wait_model.pkl")
    return {"best_model": best_model_name, "comparison": results, "metrics": best}


def train_price_model(df: pd.DataFrame) -> dict[str, Any]:
    X = df[FEATURE_COLUMNS]
    y = df[PRICE_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model_names = ["linear_regression", "decision_tree", "random_forest"]
    results = [
        evaluate_regression_model(name, X_train, X_test, y_train, y_test)
        for name in model_names
    ]
    best = max(results, key=lambda item: (item["r2"], -item["rmse"]))
    best_model_name = best["model"]
    best_model = build_regression_model(best_model_name, PRICE_TARGET)
    best_model.fit(X_train, y_train)

    ensure_model_dirs()
    joblib.dump(best_model, MODEL_DIR / "price_model.pkl")
    joblib.dump(best_model, LEGACY_MODEL_DIR / "price_model.pkl")
    joblib.dump(best_model, PROJECT_MODEL_DIR / "price_model.pkl")
    return {"best_model": best_model_name, "comparison": results, "metrics": best}


def save_summary(metrics: dict[str, Any]) -> None:
    ensure_model_dirs()
    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (LEGACY_MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (PROJECT_MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train_all_models() -> dict[str, Any]:
    df = load_dataset()
    classification = train_availability_model(df)
    wait = train_wait_model(df)
    price = train_price_model(df)

    summary = {
        "data_source": str(DATASET_PATH),
        "synthetic_demo_data_notice": "This project uses a demo/synthetic dataset of Thunderbird 350X spare-part records for prototype ML work. Predictions are estimates and do not represent guaranteed live inventory, prices, or waiting times.",
        "classification": classification,
        "wait_time_regression": wait,
        "price_regression": price,
    }
    save_summary(summary)
    return summary


def predict_example() -> None:
    df = load_dataset()
    sample = df.iloc[0].to_dict()
    availability_model = joblib.load(MODEL_DIR / "availability_model.pkl")
    wait_model = joblib.load(MODEL_DIR / "wait_model.pkl")
    price_model = joblib.load(MODEL_DIR / "price_model.pkl")

    feature_row = {
        "Bike model": sample["Bike model"],
        "Manufacturing year": sample["Manufacturing year"],
        "Part name": sample["Part name"],
        "Part category": sample["Part category"],
        "Supplier type": sample["Supplier type"],
        "City": sample["City"],
        "Garage type": sample["Garage type"],
        "Part condition": sample["Part condition"],
        "Service priority": sample["Service priority"],
        "record_month": pd.to_datetime(sample["Record date"]).month,
        "record_year": pd.to_datetime(sample["Record date"]).year,
    }

    availability = availability_model.predict(pd.DataFrame([feature_row]))[0]
    wait = wait_model.predict(pd.DataFrame([feature_row]))[0]
    price = price_model.predict(pd.DataFrame([feature_row]))[0]
    print({
        "availability": availability,
        "estimated_wait_days": round(float(wait), 2),
        "estimated_price_inr": round(float(price), 2),
    })


if __name__ == "__main__":
    summary = train_all_models()
    print("Availability model:", summary["classification"]["best_model"])
    print("Wait model:", summary["wait_time_regression"]["best_model"])
    print("Price model:", summary["price_regression"]["best_model"])
    print("Metrics file:", MODEL_DIR / "metrics.json")
    print("Saved model files:")
    for path in [MODEL_DIR / "availability_model.pkl", MODEL_DIR / "wait_model.pkl", MODEL_DIR / "price_model.pkl"]:
        print(path, path.exists())
    predict_example()
