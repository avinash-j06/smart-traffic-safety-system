import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

PROCESSED = "data/processed"
MODELS    = "models"
os.makedirs(MODELS, exist_ok=True)

# ── helper: print a nice result ───────────────────────────────
def evaluate(name, y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc*100:.1f}%")
    print(f"\n  Detailed report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return acc


# ════════════════════════════════════════════════
# MODEL 1 — Traffic Congestion Predictor
# Target: congestion_level  (0=free, 1=moderate, 2=heavy)
# ════════════════════════════════════════════════
print("\n" + "="*50)
print("MODEL 1 — Traffic Congestion Predictor")
print("="*50)

df = pd.read_csv(f"{PROCESSED}/traffic_clean.csv")

FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "is_rush_hour", "vehicle_count", "avg_speed_kmph",
    "weather_encoded", "is_holiday"
]
TARGET = "congestion_level"

X = df[FEATURES]
y = df[TARGET]

print(f"\n  Training samples : {len(X):,}")
print(f"  Features used    : {FEATURES}")
print(f"  Target classes   : {sorted(y.unique())}  (0=free 1=moderate 2=heavy)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model1 = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss",
    verbosity=0
)
print("\n  Training XGBoost... ", end="")
model1.fit(X_train, y_train)
print("done!")

y_pred = model1.predict(X_test)
acc1   = evaluate("Traffic model", y_test, y_pred)

# Save model + feature list together
joblib.dump({"model": model1, "features": FEATURES}, f"{MODELS}/traffic_model.pkl")
print(f"  Saved → models/traffic_model.pkl")


# ════════════════════════════════════════════════
# MODEL 2 — Accident Severity Predictor
# Target: severity_encoded  (0=Minor 1=Moderate 2=Severe 3=Fatal)
# ════════════════════════════════════════════════
print("\n" + "="*50)
print("MODEL 2 — Accident Severity Predictor")
print("="*50)

df = pd.read_csv(f"{PROCESSED}/accident_clean.csv")

FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "is_night", "is_rush_hour",
    "vehicles_involved", "speed_limit",
    "weather_encoded", "road_cond_encoded",
    "light_encoded", "road_type_encoded"
]
TARGET = "severity_encoded"

X = df[FEATURES]
y = df[TARGET]

print(f"\n  Training samples : {len(X):,}")
print(f"  Features used    : {FEATURES}")
print(f"  Target classes   : {sorted(y.unique())}  (0=Minor 1=Moderate 2=Severe 3=Fatal)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
print("\n  Training Random Forest... ", end="")
model2.fit(X_train, y_train)
print("done!")

y_pred = model2.predict(X_test)
acc2   = evaluate("Accident model", y_test, y_pred)

joblib.dump({"model": model2, "features": FEATURES}, f"{MODELS}/accident_model.pkl")
print(f"  Saved → models/accident_model.pkl")

# Feature importance
fi = pd.Series(model2.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n  Top 5 important features:")
print(fi.head(5).to_string())


# ════════════════════════════════════════════════
# MODEL 3 — Crime Risk Predictor
# Target: high_risk  (0=low risk, 1=high risk)
# ════════════════════════════════════════════════
print("\n" + "="*50)
print("MODEL 3 — Crime Risk Predictor")
print("="*50)

df = pd.read_csv(f"{PROCESSED}/crime_clean.csv")

FEATURES = [
    "hour", "day_of_week", "month",
    "is_night", "is_weekend", "is_rush_hour",
    "crime_type_encoded"
]
TARGET = "high_risk"

X = df[FEATURES]
y = df[TARGET]

print(f"\n  Training samples : {len(X):,}")
print(f"  Features used    : {FEATURES}")
print(f"  Target classes   : {sorted(y.unique())}  (0=low risk  1=high risk)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model3 = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
print("\n  Training Gradient Boosting... ", end="")
model3.fit(X_train, y_train)
print("done!")

y_pred = model3.predict(X_test)
acc3   = evaluate("Crime model", y_test, y_pred)

joblib.dump({"model": model3, "features": FEATURES}, f"{MODELS}/crime_model.pkl")
print(f"  Saved → models/crime_model.pkl")


# ════════════════════════════════════════════════
# Final Summary
# ════════════════════════════════════════════════
print("\n" + "="*50)
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("="*50)
print(f"  Traffic congestion model  : {acc1*100:.1f}% accuracy")
print(f"  Accident severity model   : {acc2*100:.1f}% accuracy")
print(f"  Crime risk model          : {acc3*100:.1f}% accuracy")
print("\nSaved files in models/:")
for f in os.listdir(MODELS):
    size = os.path.getsize(f"{MODELS}/{f}") / 1024
    print(f"  {f:35s}  {size:.1f} KB")
print("="*50)
print("\nNext step: Build the Flask API!")