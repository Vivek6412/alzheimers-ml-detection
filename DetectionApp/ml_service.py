import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Constants for file paths
ARTIFACT_DIR = "model_artifacts"
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.joblib")
PCA_PATH = os.path.join(ARTIFACT_DIR, "pca.joblib")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "voting_model.joblib")
DATA_CACHE_PATH = os.path.join(ARTIFACT_DIR, "processed_data.joblib")

if not os.path.exists(ARTIFACT_DIR):
    os.makedirs(ARTIFACT_DIR)

def calculate_metrics_list(y_test, predict):
    a = round(accuracy_score(y_test, predict) * 100, 3)
    p = round(precision_score(y_test, predict, average='macro') * 100, 3)
    r = round(recall_score(y_test, predict, average='macro') * 100, 3)
    f = round(f1_score(y_test, predict, average='macro') * 100, 3)
    return [a, p, r, f]

def get_cognitive_features(X, n_components=25, n_neighbors=10):
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(X)
    return pca, transformed_data

def process_dataset(csv_path):
    dataset = pd.read_csv(csv_path)
    dataset.fillna(0, inplace=True)
    
    Y = dataset['Diagnosis'].ravel()
    # Dropping columns based on original logic
    X_raw = dataset.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis=1)
    feature_names = X_raw.columns.tolist()
    X = X_raw.values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Shuffle logic from original code
    indices = np.arange(X_scaled.shape[0])
    np.random.shuffle(indices)
    X_scaled = X_scaled[indices]
    Y = Y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2)
    
    # Original code had a weird load from model/data.npy that overrode the split
    # To keep behavioral parity with the "working" state, we check if it exists
    if os.path.exists("model/data.npy"):
        try:
            data_npy = np.load("model/data.npy", allow_pickle=True)
            X_train, X_test, y_train, y_test = data_npy
        except:
            pass

    pca, X_train_pca = get_cognitive_features(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Save artifacts
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)
    joblib.dump((X_train_pca, X_test_pca, y_train, y_test), DATA_CACHE_PATH)
    
    report = {
        "total_records": X.shape[0],
        "before_pca": X.shape[1],
        "after_pca": X_train_pca.shape[1],
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0]
    }
    return report

def train_ensemble():
    if not os.path.exists(DATA_CACHE_PATH):
        raise FileNotFoundError("Processed data not found. Please run feature extraction first.")
    
    X_train, X_test, y_train, y_test = joblib.load(DATA_CACHE_PATH)
    
    results = {}
    
    # Baseline KNN
    knn_base = KNeighborsClassifier(n_neighbors=3)
    knn_base.fit(X_train, y_train)
    pred_base = knn_base.predict(X_test)
    results['Non-Ensemble'] = calculate_metrics_list(y_test, pred_base)
    
    # Ensemble
    estimators = [
        ('knn', KNeighborsClassifier(n_neighbors=3)),
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('nb', GaussianNB()),
        ('lr', LogisticRegression()),
        ('ada', AdaBoostClassifier()),
        ('xg', XGBClassifier())
    ]
    vt = VotingClassifier(estimators=estimators)
    vt.fit(X_train, y_train)
    pred_ensemble = vt.predict(X_test)
    results['Voting Based Ensemble'] = calculate_metrics_list(y_test, pred_ensemble)
    
    # Save the ensemble model
    joblib.dump(vt, MODEL_PATH)
    
    # Generate Graph
    plt.figure(figsize=(6, 3))
    df_plot = pd.DataFrame([
        ['Non-Ensemble', 'Accuracy', results['Non-Ensemble'][0]],
        ['Non-Ensemble', 'Precision', results['Non-Ensemble'][1]],
        ['Non-Ensemble', 'Recall', results['Non-Ensemble'][2]],
        ['Non-Ensemble', 'FSCORE', results['Non-Ensemble'][3]],
        ['Voting Based Ensemble', 'Accuracy', results['Voting Based Ensemble'][0]],
        ['Voting Based Ensemble', 'Precision', results['Voting Based Ensemble'][1]],
        ['Voting Based Ensemble', 'Recall', results['Voting Based Ensemble'][2]],
        ['Voting Based Ensemble', 'FSCORE', results['Voting Based Ensemble'][3]],
    ], columns=['Algorithms', 'Parameters', 'Value'])
    
    # Compatibility with older pandas/matplotlib in the environment
    import seaborn as sns
    plt.title("All Algorithms Performance Graph")
    sns.barplot(x='Parameters', y='Value', hue='Algorithms', data=df_plot)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    return results, img_b64

def predict_data(test_csv_path):
    if not all(os.path.exists(p) for p in [SCALER_PATH, PCA_PATH, MODEL_PATH]):
        raise FileNotFoundError("Model artifacts missing. Please train the model first.")
        
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    model = joblib.load(MODEL_PATH)
    
    testdata_raw = pd.read_csv(test_csv_path)
    testdata_raw.fillna(0, inplace=True)
    
    # Keep original values for output display
    original_values = testdata_raw.values.tolist()
    
    # Drop non-feature columns
    X_test = testdata_raw.drop(['PatientID', 'DoctorInCharge'], axis=1).values
    
    # Pipeline
    X_scaled = scaler.transform(X_test)
    X_pca = pca.transform(X_scaled)
    predictions = model.predict(X_pca)
    
    return original_values, predictions.tolist()
