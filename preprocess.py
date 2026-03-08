"""
preprocess.py
-------------
Loads the Heart Disease (Cleveland) dataset from UCI ML repo,
cleans it, encodes features, and saves train/test splits to data/.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target",
]

print("Loading Heart Disease dataset …")
try:
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns.")
except Exception as exc:
    print(f"Error loading data: {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
df.dropna(inplace=True)

# Binarise target: 0 = no disease, 1 = disease (original values 1-4)
df["target"] = (df["target"] > 0).astype(int)

# Cast categorical columns that arrived as float
for col in ["ca", "thal"]:
    df[col] = df[col].astype(int)

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df  = pd.concat([X_test,  y_test],  axis=1)

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv",   index=False)

print(f"  Train set: {len(train_df)} rows  →  data/train.csv")
print(f"  Test  set: {len(test_df)}  rows  →  data/test.csv")
print("Preprocessing complete.")
