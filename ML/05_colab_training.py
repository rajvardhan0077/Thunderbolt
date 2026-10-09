from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

DATASET_PATH = Path('/content/royal_enfield_thunderbird_350x_parts_dataset.csv')
OUTPUT_DIR = Path('/content/models')

def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df['Record date'] = pd.to_datetime(df['Record date'], errors='coerce')
    df['record_month'] = df['Record date'].dt.month.fillna(1)
    df['record_year'] = df['Record date'].dt.year.fillna(2025)
    return df

FEATURE_COLUMNS = [
    'Bike model',
    'Manufacturing year',
    'Part name',
    'Part category',
    'Supplier type',
    'City',
    'Garage type',
    'Part condition',
    'Service priority',
    'record_month',
    'record_year',
]
NUMERIC_COLUMNS = ['Manufacturing year', 'record_month', 'record_year']


def build_preprocessor():
    categorical = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]
    return ColumnTransformer(
        transformers=[
            ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), [c for c in FEATURE_COLUMNS if c in NUMERIC_COLUMNS]),
            ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical),
        ]
    )


def build_availability_model(name):
    if name == 'logistic_regression':
        estimator = LogisticRegression(max_iter=2000)
    elif name == 'decision_tree':
        estimator = DecisionTreeClassifier(random_state=42)
    elif name == 'random_forest':
        estimator = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced_subsample')
    else:
        raise ValueError(name)
    return Pipeline([('preprocessor', build_preprocessor()), ('classifier', estimator)])


def build_regression_model(name):
    if name == 'linear_regression':
        estimator = LinearRegression()
    elif name == 'decision_tree':
        estimator = DecisionTreeRegressor(random_state=42)
    elif name == 'random_forest':
        estimator = RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        raise ValueError(name)
    return Pipeline([('preprocessor', build_preprocessor()), ('regressor', estimator)])


def evaluate_availability(name, X_train, X_test, y_train, y_test):
    model = build_availability_model(name)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        'model': name,
        'accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'precision_weighted': round(float(precision_score(y_test, pred, average='weighted', zero_division=0)), 4),
        'recall_weighted': round(float(recall_score(y_test, pred, average='weighted', zero_division=0)), 4),
        'f1_weighted': round(float((2 * precision_score(y_test, pred, average='weighted', zero_division=0) * recall_score(y_test, pred, average='weighted', zero_division=0)) / (precision_score(y_test, pred, average='weighted', zero_division=0) + recall_score(y_test, pred, average='weighted', zero_division=0))) if (precision_score(y_test, pred, average='weighted', zero_division=0) + recall_score(y_test, pred, average='weighted', zero_division=0)) else 0, 4),
        'classification_report': classification_report(y_test, pred, output_dict=True, zero_division=0),
    }
    return metrics


def evaluate_regression(name, X_train, X_test, y_train, y_test):
    model = build_regression_model(name)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)
    return {
        'model': name,
        'mae': round(float(mae), 4),
        'mse': round(float(mse), 4),
        'rmse': round(float(rmse), 4),
        'r2': round(float(r2), 4),
    }


def train_models():
    df = load_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLUMNS]
    y_class = df['Availability status']
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
    class_results = [evaluate_availability(name, X_train, X_test, y_train, y_test) for name in ['logistic_regression', 'decision_tree', 'random_forest']]
    best_classification = max(class_results, key=lambda item: (item['f1_weighted'], item['accuracy']))
    best_class_model = build_availability_model(best_classification['model'])
    best_class_model.fit(X_train, y_train)
    joblib.dump(best_class_model, OUTPUT_DIR / 'availability_model.pkl')

    y_wait = df['Estimated wait days']
    X_wait_train, X_wait_test, y_wait_train, y_wait_test = train_test_split(X, y_wait, test_size=0.2, random_state=42)
    wait_results = [evaluate_regression(name, X_wait_train, X_wait_test, y_wait_train, y_wait_test) for name in ['linear_regression', 'decision_tree', 'random_forest']]
    best_wait = max(wait_results, key=lambda item: (item['r2'], -item['rmse']))
    best_wait_model = build_regression_model(best_wait['model'])
    best_wait_model.fit(X_wait_train, y_wait_train)
    joblib.dump(best_wait_model, OUTPUT_DIR / 'wait_model.pkl')

    y_price = df['Estimated price (INR)']
    X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)
    price_results = [evaluate_regression(name, X_price_train, X_price_test, y_price_train, y_price_test) for name in ['linear_regression', 'decision_tree', 'random_forest']]
    best_price = max(price_results, key=lambda item: (item['r2'], -item['rmse']))
    best_price_model = build_regression_model(best_price['model'])
    best_price_model.fit(X_price_train, y_price_train)
    joblib.dump(best_price_model, OUTPUT_DIR / 'price_model.pkl')

    metrics = {
        'classification': {'best_model': best_classification, 'comparison': class_results},
        'waiting_time_regression': {'best_model': best_wait, 'comparison': wait_results},
        'price_regression': {'best_model': best_price, 'comparison': price_results},
    }
    (OUTPUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('Saved models to', OUTPUT_DIR)
    return metrics


if __name__ == '__main__':
    train_models()
