# Smart Garage & Spare Parts Assistance System for Royal Enfield Thunderbird 350X

An engineering degree college project developed for Third-Year Artificial Intelligence & Data Science Engineering.  
This system addresses maintenance, part sourcing, transit delay estimation, and workshop coordination for the discontinued **Royal Enfield Thunderbird 350X** motorcycle.

---

## 1. Project Overview

The **Royal Enfield Thunderbird 350X** is a modern-classic urban cruiser motorcycle that was discontinued by Royal Enfield. Because official factory assembly has ended, sourcing genuine OEM replacement spares, estimating shipping delays across different cities, and finding specialized mechanics poses a significant challenge for riders.

This project delivers a complete, end-to-end assistance platform that combines:
1. **Machine Learning Predictive Analytics:** Predicts part availability status (Available, Limited, Discontinued), forecasts waiting periods in transit days, and estimates replacement costs in Indian Rupees (INR).
2. **Transparent Multi-Factor Garage Matching:** Ranks authorized and specialist motorcycle garages using an explainable mathematical scoring formula.
3. **Role-Based Web Application:** Provides a dedicated **User (Rider) Portal** and **Garage (Workshop) Portal** with session-based authentication, request submission, and live repair status tracking.
4. **Local SQLite Persistence:** Safely stores accounts, spare-parts inventory, and service request workflows.

---

## 2. Key Features

- **Dual-Portal Authentication (`/login`):**
  - Single login selection landing page offering explicit **USER LOGIN** and **GARAGE LOGIN** options.
  - Role-based backend route guards (`@user_required`, `@garage_required`) preventing privilege escalation.
  - Secure password hashing using `werkzeug.security`.
- **Spare Parts Search & Filter (`/parts`):**
  - Search catalog with multi-filters: Part Name, Category, Availability Status, City, Condition (New, OEM Old Stock, Used), Max Price, and Max Wait Days.
  - Direct "Book Fitting" button linking parts to the service request form.
- **AI Prediction Engine (`/predict` & `/api/predict-...`):**
  - Interactive web form & REST API endpoints for predicting stock availability, transit waiting days, and replacement pricing.
  - Provides confidence scores and transparent academic disclosures.
- **Garage Recommendations (`/recommendations`):**
  - Ranks workshops using a deterministic 5-factor equation suitable for viva defense.
- **End-to-End Service Request Workflow:**
  - Riders create requests with tracking IDs (e.g., `SR-2026-0001`).
  - Garages view incoming orders, accept/reject, update status (`Pending`, `Accepted`, `In Progress`, `Completed`, `Cancelled`), and post mechanic notes.
- **Examiner Evaluation Dashboard (`/ml-metrics`):**
  - Real-time display of empirical model comparison benchmarks, confusion matrices, and regression metrics.

---

## 3. Technology Stack

- **Backend:** Python 3.11 / 3.14, Flask 3.0+, Flask-CORS
- **Database:** SQLite 3 (parameterized queries, relational foreign keys)
- **Machine Learning & Preprocessing:** scikit-learn, pandas, NumPy, joblib
- **Frontend:** Responsive Vanilla HTML5, CSS3, JavaScript (Automotive Charcoal & Deep RE Red theme)
- **Automated Testing:** pytest (20 automated end-to-end tests)
- **Security:** Werkzeug password hashing, Flask session cookies

---

## 4. Project Directory Structure

```text
Thunderbird350X_Project/
├── app.py                     # Root execution entry point
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation & viva guide
├── .gitignore                 # Safe repository ignore rules
├── Backend/
│   ├── app.py                 # Core Flask application & REST API routes
│   ├── database.py            # SQLite schema, tables, and CRUD operations
│   ├── requirements.txt       # Backend dependency manifest
│   └── data/
│       ├── thunderbird_garage.db  # Local SQLite database (auto-seeded)
│       └── service_requests.json  # Legacy JSON storage backup
├── Dataset/
│   └── royal_enfield_thunderbird_350x_parts_dataset.csv  # 500 academic records
├── ML/
│   ├── 01_data_inspection.py           # Dataset inspection script
│   ├── 02_preprocessing_and_modeling.py # ML training & evaluation pipeline
│   ├── 03_model_comparison.ipynb       # Jupyter notebook comparison
│   ├── 04_train_final_models.py        # Model retraining launcher
│   ├── 05_colab_training.py            # Colab training backup
│   └── models/
│       ├── availability_model.pkl      # Decision Tree classifier
│       ├── wait_model.pkl              # Random Forest regressor
│       ├── price_model.pkl             # Linear Regression model
│       └── metrics.json                # Empirical evaluation results
├── EDA/
│   ├── 01_data_inspection.py
│   └── 01_data_inspection_colab.py
├── templates/                 # Web application Jinja2 templates
│   ├── base.html              # Layout, navbar, disclaimer banner, footer
│   ├── index.html             # Home landing page with metrics
│   ├── login_select.html      # Two-choice portal selection (/login)
│   ├── login_user.html        # Rider login form
│   ├── login_garage.html      # Workshop login form
│   ├── register.html          # User and garage registration form
│   ├── user_dashboard.html    # Rider dashboard & request status
│   ├── garage_dashboard.html  # Workshop management & status updater
│   ├── parts_search.html      # Multi-filter spare parts catalog
│   ├── predict.html           # Interactive AI prediction tool
│   ├── recommendations.html   # Transparent garage ranking
│   ├── new_request.html       # Service order submission form
│   ├── service_requests.html  # User repair tracking timeline
│   ├── ml_metrics.html        # Examiner viva metrics dashboard
│   └── profile.html           # User account profile
├── static/
│   ├── css/
│   │   └── styles.css         # Automotive styling design system
│   └── js/
│       └── main.js            # Client-side AJAX prediction & helpers
├── tests/
│   └── test_pipeline.py       # 20 automated pytest unit & integration tests
├── docs/                      # Technical documentation files
│   ├── api_documentation.md
│   ├── architecture.md
│   ├── viva_questions.md
│   └── ml_methodology.md
└── Flutter_App/               # Preserved mobile app assets
```

---

## 5. Dataset Location & Academic Limitations

- **File Path:** `Dataset/royal_enfield_thunderbird_350x_parts_dataset.csv`
- **Verified Dimensions:** 500 records &bull; 13 columns &bull; 0 missing values &bull; 0 duplicate rows
- **Columns:**
  `Bike model`, `Manufacturing year`, `Part name`, `Part category`, `Availability status`, `Estimated wait days`, `Estimated price (INR)`, `Supplier type`, `City`, `Garage type`, `Part condition`, `Service priority`, `Record date`.

### Important Academic Data Disclosure
> **Notice:** The dataset contains synthetic academic data created for university project demonstrations. It does **not** represent verified live Royal Enfield factory inventory, real-time distributor warehouse stock, or guaranteed market quotations. All predictions, prices, and transit days must be treated as educational estimates.

---

## 6. Machine Learning Methodology & Empirical Results

The pipeline uses `sklearn.compose.ColumnTransformer` with `OneHotEncoder` and `SimpleImputer` inside reproducible pipelines (`random_state=42`, 80/20 train/test split).

### A. Availability Classification (3 Classes: Available, Limited, Discontinued)
Evaluated with Accuracy, Weighted Precision, Weighted Recall, Weighted F1, and Confusion Matrix.

| Algorithm | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree Classifier** | **37.0%** | **39.83%** | **37.0%** | **37.50%** | **Selected (Best F1)** |
| Logistic Regression | 41.0% | 37.18% | 41.0% | 37.46% | Evaluated |
| Random Forest Classifier | 38.0% | 27.91% | 38.0% | 30.84% | Evaluated |

### B. Waiting Time Regression (Days)
Evaluated with Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R²:

| Algorithm | MAE | RMSE | R² Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **9.83 days** | **13.68 days** | **-0.118** | **Selected (Lowest MAE)** |
| Linear Regression | 10.86 days | 14.77 days | -0.305 | Evaluated |
| Decision Tree Regressor | 12.38 days | 19.72 days | -1.325 | Evaluated |

### C. Spare Part Price Regression (INR)
Evaluated with MAE, RMSE, and R²:

| Algorithm | MAE | RMSE | R² Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | **₹619.55** | **₹967.92** | **0.8610** | **Selected (Best R²)** |
| Random Forest Regressor | ₹653.93 | ₹1098.38 | 0.8210 | Evaluated |
| Decision Tree Regressor | ₹963.90 | ₹1670.53 | 0.5859 | Evaluated |

---

## 7. Transparent Garage Recommendation Formula (Viva Defense)

To ensure full transparency during viva examinations, garage rankings avoid black-box obscurity. Each candidate is assigned an explainable score out of 100:

$$\text{Score} = 100 \times \left(0.40 \cdot S_{\text{avail}} + 0.25 \cdot S_{\text{wait}} + 0.20 \cdot S_{\text{price}} + 0.10 \cdot S_{\text{garage}} + 0.05 \cdot S_{\text{supplier}}\right)$$

- **$S_{\text{avail}}$:** Stock availability weight ($1.0$ for Available, $0.6$ for Limited, $0.2$ for Discontinued).
- **$S_{\text{wait}}$:** Reciprocal normalized transit delay $\max(0, 1 - \frac{\text{Wait}}{\text{MaxWait}})$.
- **$S_{\text{price}}$:** Reciprocal normalized component cost $\max(0, 1 - \frac{\text{Price}}{\text{MaxPrice}})$.
- **$S_{\text{garage}}$:** Workshop category bonus ($0.7$ for Royal Enfield Specialists, $0.5$ for General).
- **$S_{\text{supplier}}$:** Supplier reliability bonus ($0.9$ for OEM Authorized, $0.6$ for Third-party).

---

## 8. Installation & Setup on Windows

### Prerequisites
- Windows 10 or 11
- Python 3.11+ (or Python 3.14) installed and added to `PATH`
- VS Code or Windows PowerShell

### Step 1: Clone / Open Project Folder
Open PowerShell in the project directory:
```powershell
cd D:\Docs\Projects\Thunderbird350X_Project
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Packages
```powershell
pip install -r requirements.txt
```

### Step 4: Train Machine Learning Models (Optional - Pretrained models included)
If you wish to retrain all models from scratch:
```powershell
python ML/02_preprocessing_and_modeling.py
```

### Step 5: Initialize the Database (Auto-initialized on startup)
The SQLite database `Backend/data/thunderbird_garage.db` is initialized and seeded automatically whenever the app starts. To test manually:
```powershell
python -c "import sys; sys.path.insert(0, 'Backend'); import database; database.seed_db()"
```

---

## 9. How to Run the Application

Start the local web server with:
```powershell
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### Pre-Configured Demo Accounts for Testing & Viva Demonstration

| Role | Portal Link | Username | Password | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Motorcycle Rider** | [http://127.0.0.1:5000/login/user](http://127.0.0.1:5000/login/user) | `user_demo` | `password123` | Part searching, AI predictions, order booking |
| **Service Garage** | [http://127.0.0.1:5000/login/garage](http://127.0.0.1:5000/login/garage) | `garage_demo` | `garage123` | Workshop order management, status updates |

---

## 10. Automated Testing

Run the automated test suite with pytest:
```powershell
pytest tests/test_pipeline.py -v
```

All **20 tests** validate:
- Dataset integrity & column presence
- SQLite database tables, seeding, and foreign keys
- Home page, login selection, and auth redirection
- Role-based access control (User vs. Garage boundaries)
- Trained model loading & inference output
- Availability, waiting-time, and price prediction APIs
- Error handling on invalid/empty JSON payloads
- Spare parts multi-filter catalog queries
- End-to-end service request lifecycle (create &rarr; accept &rarr; in-progress &rarr; complete)
- Unauthorized cross-garage status update prevention

---

## 11. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status and model load verification |
| `POST` | `/api/predict-availability` | Predicts availability category (`Available`, `Limited`, `Discontinued`) |
| `POST` | `/api/predict-wait` | Forecasts estimated transit waiting time in days |
| `POST` | `/api/predict-price` | Predicts expected replacement price in INR |
| `POST` | `/api/predict-part-details` | Combined prediction with confidence and part summary |
| `POST` | `/api/find-garages` | Ranks matching workshops with recommendation scores |
| `POST` | `/api/service-request` | Submits a new customer repair request |
| `GET` | `/api/service-requests` | Retrieves service requests for the active user/garage |
| `GET` | `/api/ml-metrics` | Fetches JSON benchmark metrics from `metrics.json` |

---

## 12. Known Limitations & Future Scope

### Known Limitations
1. **Synthetic Training Records:** Sourced from 500 simulated records; not connected to live OEM ERP or SAP inventory systems.
2. **Discontinued Model Specificity:** Tuned specifically for Royal Enfield Thunderbird 350X components and may not directly map to newer J-platform models (Meteor 350, Hunter 350).
3. **Local Deployment:** Configured for local SQLite operation; production deployment would use PostgreSQL with WSGI (Gunicorn).

### Future Scope
1. Real-time integration with local distributor WhatsApp APIs or automated SMS status alerts.
2. OCR scanner for physical Royal Enfield part catalogues and bills.
3. Geo-location mapping (OpenStreetMap / Leaflet) for distance-based workshop routing.

---

## 13. Git Commands for Repository Update

To commit and push your completed project changes to your GitHub repository:

```bash
# 1. Check current status
git status

# 2. Add modified and newly created project files
git add app.py requirements.txt README.md .gitignore Backend/ ML/ static/ templates/ tests/

# 3. Commit the changes
git commit -m "Complete Thunderbird 350X Smart Garage web app, ML models, SQLite auth, and pytest suite"

# 4. Push to your main branch
git push origin main
```
*(GitHub repository: https://github.com/rajvardhan0077/Thunderbird-350X-Smart-Garage)*
