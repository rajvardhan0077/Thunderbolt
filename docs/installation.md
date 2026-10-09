# Installation

## ML environment

A non-blocked Python environment is required. This project was prepared to run in a clean Python 3.11 environment or Google Colab.

### Windows local attempt

This machine currently blocks pandas/scipy DLL imports due to an Application Control policy, so direct local model training is blocked.

### Colab route

Upload the project folder or the `Dataset/` CSV and run the training script.

```python
!pip install pandas numpy scikit-learn joblib flask flask-cors
```

Then run:

```python
!python ML/05_colab_training.py
```

## Backend installation

```bash
cd Backend
pip install -r requirements.txt
python app.py
```

## Flutter setup

```bash
cd Flutter_App
flutter pub get
flutter run
```
