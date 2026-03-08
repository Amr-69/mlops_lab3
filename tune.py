"""
tune.py
-------
Uses MLflow nested runs to perform a grid search over two
hyperparameters (n_estimators, max_depth) for RandomForestClassifier.
The best run's model is saved as model_rf_best.joblib.
"""

import itertools
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
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
# Hyperparameter grid  (≥ 2 params as required)
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [None, 5, 10],
}

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
mlflow.set_experiment("Heart Disease Classification")

best_auc   = -1.0
best_model = None
best_params = {}

combos = list(itertools.product(PARAM_GRID["n_estimators"], PARAM_GRID["max_depth"]))

with mlflow.start_run(run_name="RandomForest Hyperparameter Tuning"):
    mlflow.set_tag("model_type", "Random Forest Grid Search")
    mlflow.log_param("n_estimators_values", str(PARAM_GRID["n_estimators"]))
    mlflow.log_param("max_depth_values",    str(PARAM_GRID["max_depth"]))

    for n_est, max_d in combos:
        run_name = f"RF_n{n_est}_d{max_d if max_d else 'None'}"

        with mlflow.start_run(run_name=run_name, nested=True):
            params = {
                "n_estimators": n_est,
                "max_depth":    max_d,
                "random_state": 42,
                "n_jobs":       -1,
            }
            mlflow.log_params(params)
            mlflow.set_tag("model_type", "Random Forest")

            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)

            y_pred  = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy":  accuracy_score(y_test, y_pred),
                "auc":       roc_auc_score(y_test, y_proba),
                "f1":        f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall":    recall_score(y_test, y_pred),
            }
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(rf, "random_forest_model")
            print(f"  {run_name} → AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}")

            if metrics["auc"] > best_auc:
                best_auc    = metrics["auc"]
                best_model  = rf
                best_params = params

    # Log best result to parent run
    mlflow.log_metric("best_auc", best_auc)
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

# Save best model to disk so DVC can track it
joblib.dump(best_model, "model_rf_best.joblib")
print(f"\nBest RF → AUC={best_auc:.4f}  params={best_params}")
print("Tuning complete.")
