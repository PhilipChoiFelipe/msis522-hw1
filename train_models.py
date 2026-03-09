"""
MSIS 522 HW1 — Model Training Script
Household Electric Power Consumption (Regression Task)
Target: Global_active_power (kW)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import shap
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH = "../hw1_data/household_power_consumption.txt"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Load & Pre-process ─────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, sep=";", na_values=["?"], low_memory=False)

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
)
df.drop(columns=["Date", "Time"], inplace=True)

num_cols = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df.dropna(inplace=True)
df.sort_values("Datetime", inplace=True)
df.reset_index(drop=True, inplace=True)

# Time features
df["hour"]        = df["Datetime"].dt.hour
df["day_of_week"] = df["Datetime"].dt.dayofweek
df["month"]       = df["Datetime"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

print(f"Full dataset shape: {df.shape}")

# Sample 50k rows for tractable training
SAMPLE_N = 50_000
df_sample = df.sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)
df_sample.to_parquet(os.path.join(MODELS_DIR, "sample_data.parquet"), index=False)
print(f"Saved sample ({SAMPLE_N} rows) → models/sample_data.parquet")

# ── 2. Features / Target ──────────────────────────────────────────────────
FEATURES = [
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "hour", "day_of_week", "month", "is_weekend",
]
TARGET = "Global_active_power"

X = df_sample[FEATURES]
y = df_sample[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

joblib.dump(
    {"X_train": X_train, "X_test": X_test,
     "y_train": y_train, "y_test": y_test},
    os.path.join(MODELS_DIR, "train_test_split.pkl"),
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Scaler for LR & MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

# ── Helper ────────────────────────────────────────────────────────────────
def reg_metrics(y_true, y_pred, name):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    print(f"  {name}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"Model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

all_metrics = []
best_params_dict = {}

# ── 2.2 Linear Regression ─────────────────────────────────────────────────
print("\n--- 2.2 Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
all_metrics.append(reg_metrics(y_test, y_pred_lr, "Linear Regression"))
joblib.dump(lr, os.path.join(MODELS_DIR, "linear_regression.pkl"))
np.save(os.path.join(MODELS_DIR, "pred_lr.npy"), y_pred_lr)
best_params_dict["Linear Regression"] = {"Note": "No hyperparameters tuned"}

# ── 2.3 Decision Tree ─────────────────────────────────────────────────────
print("\n--- 2.3 Decision Tree (GridSearchCV 5-fold) ---")
dt_grid = {"max_depth": [3, 5, 7, 10], "min_samples_leaf": [5, 10, 20, 50]}
dt_cv = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    dt_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1,
)
dt_cv.fit(X_train, y_train)
print(f"  Best: {dt_cv.best_params_}")
y_pred_dt = dt_cv.predict(X_test)
all_metrics.append(reg_metrics(y_test, y_pred_dt, "Decision Tree"))
joblib.dump(dt_cv.best_estimator_, os.path.join(MODELS_DIR, "decision_tree.pkl"))
np.save(os.path.join(MODELS_DIR, "pred_dt.npy"), y_pred_dt)
best_params_dict["Decision Tree"] = dt_cv.best_params_

# ── 2.4 Random Forest ────────────────────────────────────────────────────
print("\n--- 2.4 Random Forest (GridSearchCV 5-fold) ---")
rf_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 8]}
rf_cv = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1,
)
rf_cv.fit(X_train, y_train)
print(f"  Best: {rf_cv.best_params_}")
y_pred_rf = rf_cv.predict(X_test)
all_metrics.append(reg_metrics(y_test, y_pred_rf, "Random Forest"))
joblib.dump(rf_cv.best_estimator_, os.path.join(MODELS_DIR, "random_forest.pkl"))
np.save(os.path.join(MODELS_DIR, "pred_rf.npy"), y_pred_rf)
best_params_dict["Random Forest"] = rf_cv.best_params_

# ── 2.5 XGBoost ──────────────────────────────────────────────────────────
print("\n--- 2.5 XGBoost (GridSearchCV 5-fold) ---")
xgb_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
}
xgb_cv = GridSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0, tree_method="hist"),
    xgb_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1,
)
xgb_cv.fit(X_train, y_train)
print(f"  Best: {xgb_cv.best_params_}")
y_pred_xgb = xgb_cv.predict(X_test)
all_metrics.append(reg_metrics(y_test, y_pred_xgb, "XGBoost"))
joblib.dump(xgb_cv.best_estimator_, os.path.join(MODELS_DIR, "xgboost.pkl"))
np.save(os.path.join(MODELS_DIR, "pred_xgb.npy"), y_pred_xgb)
best_params_dict["XGBoost"] = xgb_cv.best_params_

# ── 2.6 MLP Neural Network (Keras) ───────────────────────────────────────
print("\n--- 2.6 MLP Neural Network (Keras) ---")

mlp = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(len(FEATURES),)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64,  activation="relu"),
    keras.layers.Dense(1),
])
mlp.compile(optimizer="adam", loss="mse", metrics=["mae"])
mlp.summary()

history = mlp.fit(
    X_train_scaled, y_train.values,
    validation_split=0.1,
    epochs=30,
    batch_size=512,
    verbose=1,
)
y_pred_mlp = mlp.predict(X_test_scaled, verbose=0).flatten()
all_metrics.append(reg_metrics(y_test, y_pred_mlp, "MLP Neural Network"))
mlp.save(os.path.join(MODELS_DIR, "mlp_model.h5"))
np.save(os.path.join(MODELS_DIR, "pred_mlp.npy"), y_pred_mlp)
best_params_dict["MLP Neural Network"] = {
    "architecture": "128-128-64-1",
    "optimizer": "adam",
    "epochs": 30,
    "batch_size": 512,
}

hist_json = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(os.path.join(MODELS_DIR, "mlp_history.json"), "w") as f:
    json.dump(hist_json, f)

# ── 2.7 Model Comparison ─────────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(MODELS_DIR, "model_metrics.csv"), index=False)
print("\n=== Model Comparison ===")
print(metrics_df.to_string(index=False))

with open(os.path.join(MODELS_DIR, "best_params.json"), "w") as f:
    json.dump(best_params_dict, f, default=str)

# ── 3. SHAP (XGBoost — best tree model) ──────────────────────────────────
print("\n--- 3. SHAP Analysis ---")
best_xgb = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
explainer  = shap.TreeExplainer(best_xgb)

shap_sample = X_test.sample(n=500, random_state=42).reset_index(drop=True)
shap_values = explainer.shap_values(shap_sample)

joblib.dump(explainer,  os.path.join(MODELS_DIR, "shap_explainer.pkl"))
np.save(os.path.join(MODELS_DIR, "shap_values.npy"), shap_values)
joblib.dump(shap_sample, os.path.join(MODELS_DIR, "shap_sample.pkl"))

print(f"SHAP values shape: {shap_values.shape} — saved.")

# ── Metadata ─────────────────────────────────────────────────────────────
meta = {
    "features": FEATURES,
    "target": TARGET,
    "n_total": int(df.shape[0]),
    "n_sample": SAMPLE_N,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "feature_means": {k: float(v) for k, v in X_train.mean().items()},
    "feature_stds":  {k: float(v) for k, v in X_train.std().items()},
    "feature_mins":  {k: float(v) for k, v in X_train.min().items()},
    "feature_maxs":  {k: float(v) for k, v in X_train.max().items()},
    "y_mean": float(y_train.mean()),
    "y_std":  float(y_train.std()),
}
with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f)

print("\nAll artifacts saved to ./models/")
print("Training complete!")
