# Alzheimer’s Disease Detection via Voting-Based Ensemble ML

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Django Version](https://img.shields.io/badge/django-4.2_LTS-green.svg)](https://www.djangoproject.com/)
[![Database](https://img.shields.io/badge/database-PostgreSQL-blue.svg)](https://supabase.com/)

An advanced diagnostic decision-support system designed to identify Alzheimer's disease markers using clinical and cognitive features. This project implements a high-performance **Voting-Based Ensemble** model, achieving superior predictive accuracy by combining multiple specialized machine learning algorithms.

**Live Demo:** [https://web-production-3ca46.up.railway.app/](https://web-production-3ca46.up.railway.app/)

---

## 🏗 System Architecture

The application follows a clean, decoupled architecture optimized for scalability and thread safety:

-   **Frontend:** Responsive UI built with **Tailwind CSS** and **Inter Typography**, ensuring a professional experience across desktop and mobile.
-   **Backend:** **Django 4.2 LTS** utilizing a stateless service layer for ML operations, fully compatible with modern serverless and containerized environments.
-   **Database:** **PostgreSQL (Supabase)** with **Django ORM** integration for secure, encrypted user management.
-   **ML Pipeline:** Decoupled service layer (`ml_service.py`) with artifact persistence via **Joblib**, supporting high-concurrency diagnostic requests.

---

## 🧠 Machine Learning Methodology

### 1. Data Preprocessing
-   **Missing Value Imputation:** Automated handling of null entries.
-   **Standardization:** Feature scaling via `StandardScaler` to ensure uniform algorithm weight.
-   **Cognitive Feature Selection:** Utilizes custom **Principal Component Analysis (PCA)** to isolate high-variance cognitive markers.

### 2. The Ensemble Model
Our **VotingClassifier** utilizes a "Hard Voting" strategy, aggregating predictions from:
-   **KNN & Decision Trees** (Local patterns)
-   **Random Forest & AdaBoost** (Robust bagging/boosting)
-   **XGBoost** (High-performance gradient boosting)
-   **Naive Bayes & Logistic Regression** (Probabilistic baselines)

---

## 🚀 Installation & Setup

### Prerequisites
-   Python 3.12+
-   PostgreSQL (Cloud hosting via **Supabase** recommended)

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Vivek6412/alzheimers-ml-detection.git
cd Alzheimer

# Install dependencies
pip install -r requirements.txt
```

### 2. Application Initialization (MANDATORY)
Run these commands to initialize the secure database schema and build essential tables:
```bash
# Generate and Apply Django Migrations
python manage.py makemigrations
python manage.py migrate

# Start the Development Server
python manage.py runserver 8000
```

---

## 🔒 Security Features
-   **SQL Injection Prevention:** 100% ORM-backed database interactions.
-   **Cryptographic Hashing:** User passwords are encrypted using **PBKDF2 with SHA256**.
-   **Access Control:** All diagnostic features are protected by Django’s `SessionAuthentication` and `@login_required` middleware.
-   **Cloud Parity:** Environment-based configuration for secret keys and database credentials.

---

## 📂 Project Structure
```text
├── Dataset/             # Clinical CSV data samples
├── Detection/           # Core Project Configuration
├── DetectionApp/        # Application Logic
│   ├── ml_service.py    # Isolated Machine Learning Pipeline
│   ├── models.py        # ORM Schema Definitions
│   ├── views.py         # Request Controllers (Clean & Secure)
│   └── templates/       # Modern Responsive UI (Tailwind)
├── model_artifacts/     # Persisted ML Models (joblib)
├── README.md            # Technical Documentation
└── requirements.txt     # Production Dependency Manifest
```

---
© 2026 AlzheimerCare AI. Developed with a focus on medical diagnostic accuracy and cloud-native security.
