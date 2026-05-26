# Alzheimer’s Disease Detection via Voting-Based Ensemble ML

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![Django Version](https://img.shields.io/badge/django-2.1.7-green.svg)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An advanced diagnostic decision-support system designed to identify Alzheimer's disease markers using clinical and cognitive features. 

**Live Demo:** [https://web-production-3ca46.up.railway.app/](https://web-production-3ca46.up.railway.app/)

---

## 🏗 System Architecture

The application follows a clean, decoupled architecture optimized for scalability and thread safety:

-   **Frontend:** Responsive UI built with **Tailwind CSS** and **Inter Typography**, ensuring a professional experience across desktop and mobile.
-   **Backend:** **Django 2.1.7** utilizing a stateless service layer for ML operations.
-   **Database:** **MySQL** with **Django ORM** integration for secure, encrypted user management.
-   **ML Pipeline:** Decoupled service layer (`ml_service.py`) with artifact persistence via **Joblib**.

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
-   Python 3.7.9 (Optimized for legacy TF/Keras compatibility)
-   MySQL Server 8.0+

### 1. Database Configuration
Create the application database and legacy registration table:
```sql
CREATE DATABASE alzheimer;
USE alzheimer;

CREATE TABLE register(
    username VARCHAR(50) PRIMARY KEY,
    password VARCHAR(50),
    contact_no VARCHAR(20),
    email VARCHAR(50), 
    address VARCHAR(80)
);
```

### 2. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd Alzheimer

# Install dependencies
pip install -r requirements.txt
```

### 3. Application Initialization (MANDATORY)
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

---

## 📂 Project Structure
```text
├── Dataset/             # Clinical CSV data samples
├── Detection/           # Core Project Configuration
├── DetectionApp/        # Application Logic
│   ├── ml_service.py    # Isolated Machine Learning Pipeline
│   ├── models.py        # ORM Schema Definitions
│   ├── views.py         # Request Controllers
│   └── templates/       # Modern Responsive UI
├── model_artifacts/     # Persisted Models (joblib)
└── requirements.txt     # Dependency manifest
```

---
© 2026 AlzheimerCare AI. Developed with a focus on medical diagnostic accuracy and security.
