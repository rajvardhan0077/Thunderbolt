# Architecture

This project is designed as a prototype ML + web + mobile pipeline for Thunderbird 350X spare-part assistance.

## System flow

1. User enters part details in the Flutter app.
2. Flutter sends a request to the Flask backend.
3. Flask validates the request and loads trained ML models.
4. The backend preprocesses inputs and predicts availability, wait time, and price.
5. It also ranks garages/suppliers using a transparent scoring approach.
6. Results are returned to the mobile app for display.

## Components

- Dataset: CSV-based demo dataset in the `Dataset/` folder.
- ML: `ML/02_preprocessing_and_modeling.py` and `ML/05_colab_training.py` train and persist models.
- Backend: `Backend/app.py` exposes REST endpoints.
- Flutter: `Flutter_App/` includes a simple mobile UI for searching parts and viewing results.
- Documentation: the `docs/` folder stores architecture and project notes.

## Demo data note

This project uses a demo dataset for prototype work. It does not provide real-time inventory or live pricing.
