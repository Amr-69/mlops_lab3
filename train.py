"""
train.py
--------
Trains a Logistic Regression baseline on data/train.csv,
evaluates on data/test.csv, and logs everything to MLflow.
Saves the trained model as model.joblib.
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")

X_train, y_train = train_df.drop("target", axis=1), train_df["target"]
X_test,  y_test  = test_df.drop("target",  axis=1), test_df["target"]

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
mlflow.set_experiment("Heart Disease Classification")

params = {"solver": "liblinear", "C": 1.0, "random_state": 42, "max_iter": 1000}

with mlflow.start_run(run_name="Logistic Regression Baseline"):
    mlflow.set_tag("model_type", "Logistic Regression")
    mlflow.log_params(params)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**params)),
    ])
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc":      roc_auc_score(y_test, y_proba),
        "f1":       f1_score(y_test, y_pred),
    }
    mlflow.log_metrics(metrics)
    print("  Logistic Regression →", metrics)

    # Save & log model
    joblib.dump(pipe, "model.joblib")
    mlflow.log_artifact("model.joblib")
    mlflow.sklearn.log_model(pipe, "logistic_regression_model")

print("Training complete.")
