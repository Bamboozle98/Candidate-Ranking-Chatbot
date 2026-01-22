# train_logistic_regression.py

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from joblib import dump

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CSV_PATH = r"C:\Users\cbran\PycharmProjects\8XPTuDF1AleElmm6\data\raw\potential-talents - Aspiring human resources - seeking human resources.csv"        # path to your CSV
TARGET_COL = "fit"      # name of target column
MODEL_OUT = "logreg_model.joblib"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# --------------------------------------------------
# FEATURE TYPES
# --------------------------------------------------
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# --------------------------------------------------
# MODEL (STRONG REGULARIZATION)
# --------------------------------------------------
log_reg = LogisticRegression(
    penalty="l2",
    C=0.1,               # smaller = more regularization (good for small data)
    solver="lbfgs",
    max_iter=1000
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", log_reg)
    ]
)

# --------------------------------------------------
# CROSS-VALIDATED PROBABILITIES
# --------------------------------------------------
cv = StratifiedKFold(n_splits=min(5, y.value_counts().min()), shuffle=True, random_state=42)

probs = cross_val_predict(
    pipeline,
    X,
    y,
    cv=cv,
    method="predict_proba"
)[:, 1]

# --------------------------------------------------
# METRICS (PROBABILITY-BASED)
# --------------------------------------------------
print("Log Loss:", log_loss(y, probs))
print("Brier Score:", brier_score_loss(y, probs))

# --------------------------------------------------
# FIT FINAL MODEL
# --------------------------------------------------
pipeline.fit(X, y)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------
dump(pipeline, MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")

from joblib import load
import pandas as pd

model = load("logreg_model.joblib")

new_candidate = pd.DataFrame([{
    "experience_years": 4,
    "education_level": "Bachelor",
    "skill_score": 0.66
}])

prob = model.predict_proba(new_candidate)[0, 1]
print(f"Probability of good fit: {prob:.3f}")

