# Smart Garage & Spare Parts Assistance System for Royal Enfield Thunderbird 350X

This project is a prototype AI/ML and mobile application for assisting with Royal Enfield Thunderbird 350X spare-part availability, waiting time, price estimation, and garage recommendation.

## Problem statement

Royal Enfield Thunderbird 350X is a discontinued motorcycle model, so finding compatible spare parts, estimating delays, and identifying suitable suppliers or garages can be difficult. This project demonstrates a practical AI/ML prototype that helps users search for a part, estimate availability, estimate price, and identify suitable recommendation options.

## Objective

- Search for a part by model, category, and city
- Predict whether it is Available, Limited, or Discontinued
- Estimate waiting time in days
- Estimate expected price in INR
- Rank garages and suppliers based on a transparent score
- Submit and review service requests through a mobile app

## Demo / synthetic data notice

This project uses a prototype demo dataset. Predictions are estimates only and do not represent guaranteed live inventory, live prices, or live garage availability.

## Features

- ML-based part availability classification
- Regression for waiting time and price estimation
- Transparent garage recommendation logic
- Flask REST API for backend prediction services
- Flutter mobile UI for search and submission
- Service request submission and status display

## Technology stack

- Python
- pandas
- scikit-learn
- Flask
- Flask-CORS
- joblib
- Flutter
- Dart
- Firebase (optional / configuration if added later)

## Project structure

- `Dataset/` – project dataset
- `ML/` – preprocessing and model training scripts
- `Backend/` – Flask API implementation
- `Flutter_App/` – Flutter mobile application
- `docs/` – project documentation
- `tests/` – automated tests

## System architecture

1. Flutter app collects part details.
2. Flask backend validates and preprocesses input.
3. Trained ML models predict availability, wait time, and price.
4. Garage ranking uses the dataset to present a recommended list.
5. Results and requests are shown to the user.

## Dataset description

The dataset contains synthetic/demo records for Thunderbird 350X spare parts. It includes fields such as:

- Bike model
- Manufacturing year
- Part name
- Part category
- Availability status
- Estimated wait days
- Estimated price (INR)
- Supplier type
- City
- Garage type
- Part condition
- Service priority
- Record date

## ML methodology

The code in `ML/02_preprocessing_and_modeling.py` preprocesses the dataset and trains:

- availability classification model
- wait-time regression model
- price estimation regression model

Models are trained with a fixed random seed, evaluated with real metrics, and saved as `.pkl` files.

## Algorithms used

- Logistic Regression
- Decision Tree
- Random Forest
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Evaluation metrics

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- MAE
- MSE
- RMSE
- R²

## Backend API documentation

See `docs/api_documentation.md`.

## Flutter application

The Flutter app in `Flutter_App/` provides:

- splash screen
- login/register UI
- home dashboard
- part search form
- prediction results
- garage recommendations
- service request form
- status screen

## Firebase setup

Firebase is optional and should only be used if configured in a separate setup. This project keeps ML prediction logic in the Flask API rather than forcing Firebase into the model pipeline.

## Google Maps setup

Maps integration should only be added if real location data is available. The project does not claim real-time location accuracy without an actual configuration and data source.

## Installation instructions

### Python environment

Because of a Windows Application Control policy blocking some pandas/scipy DLLs on this machine, direct local execution is blocked here. Train the model in a non-blocked environment such as Google Colab or a Linux VM.

#### In Colab

```python
!pip install pandas numpy scikit-learn joblib flask flask-cors
!python ML/05_colab_training.py
```

#### In a clean Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Backend/requirements.txt
python ML/04_train_final_models.py
```

## How to train models

```bash
cd "ML Project/Thunderbird350X_Project"
python ML/04_train_final_models.py
```

If the machine is blocked by the DLL policy, use the Colab script instead:

```python
!python ML/05_colab_training.py
```

## How to start the backend

```bash
cd "ML Project/Thunderbird350X_Project/Backend"
python app.py
```

## How to run the Flutter app

```bash
cd "ML Project/Thunderbird350X_Project/Flutter_App"
flutter pub get
flutter run
```

## Example API requests

### Availability check

```json
{
  "bike_model": "Royal Enfield Thunderbird 350X",
  "manufacturing_year": 2019,
  "part_name": "Clutch Cable",
  "part_category": "Control",
  "supplier_type": "Online Supplier",
  "city": "Pune",
  "part_condition": "New",
  "service_priority": "High"
}
```

### Garage recommendation

```json
{
  "part_name": "Clutch Cable",
  "city": "Pune"
}
```

## Limitations

- Demo/synthetic data is used
- No live inventory source is connected
- No live price feed is connected
- Recommendations are based on the local dataset rather than real-time garage status
- Model accuracy depends on the data quality and size

## Future scope

- connect live supplier APIs
- add real service request storage with Firebase
- add Google Maps distance support when a real API key is configured
- improve model pipelines with larger real-world data
- deploy the backend and app for demo testing

## Final project status

ML: partial / blocked locally due to Windows DLL application-control policy
Backend: implemented as a Flask API scaffold with the required endpoints
Flutter: implemented as an app skeleton with mobile layout and API integration hooks
Testing: provided as test code, but actual runtime verification is blocked by the local DLL policy
Documentation: created and updated in the `docs/` folder
