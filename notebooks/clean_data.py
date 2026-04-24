import pandas as pd
import numpy as np
import os

RAW       = "data/raw"
PROCESSED = "data/processed"
os.makedirs(PROCESSED, exist_ok=True)

# ── helper ────────────────────────────────────────────────────
def report(df, name):
    print(f"\n{'='*50}")
    print(f" {name}")
    print(f"{'='*50}")
    print(f"  Shape      : {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    print(f"  Null values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
    print(f"  Duplicates : {df.duplicated().sum()}")
    print(f"\n  Data types:\n{df.dtypes}")
    print(f"\n  Sample:\n{df.head(3).to_string()}")

# ════════════════════════════════════════════════
# 1. TRAFFIC DATA
# ════════════════════════════════════════════════
print("\n[1/3] Cleaning traffic data...")
df = pd.read_csv(f"{RAW}/traffic_data.csv")
report(df, "TRAFFIC — raw")

# Parse datetime
df["datetime"]    = pd.to_datetime(df["date"] + " " + df["time"])
df["month"]       = df["datetime"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["is_rush_hour"]= df["hour"].apply(lambda h: 1 if (7<=h<=9 or 17<=h<=19) else 0)

# Encode weather
weather_map = {"Clear":0, "Cloudy":1, "Rainy":2, "Foggy":3}
df["weather_encoded"] = df["weather"].map(weather_map)

# Drop raw date/time strings (we have datetime now)
df.drop(columns=["date","time"], inplace=True)

# Remove duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Removed {before - len(df)} duplicate rows")

# Clip outliers — speed can't be negative or > 120
df["avg_speed_kmph"] = df["avg_speed_kmph"].clip(0, 120)

df.to_csv(f"{PROCESSED}/traffic_clean.csv", index=False)
print(f"  Saved → data/processed/traffic_clean.csv  ({len(df):,} rows)")


# ════════════════════════════════════════════════
# 2. ACCIDENT DATA
# ════════════════════════════════════════════════
print("\n[2/3] Cleaning accident data...")
df = pd.read_csv(f"{RAW}/accident_data.csv")
report(df, "ACCIDENT — raw")

df["datetime"]     = pd.to_datetime(df["date"] + " " + df["time"])
df["month"]        = df["datetime"].dt.month
df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
df["is_night"]     = df["hour"].apply(lambda h: 1 if (h<6 or h>21) else 0)
df["is_rush_hour"] = df["hour"].apply(lambda h: 1 if (7<=h<=9 or 17<=h<=19) else 0)

# Encode categoricals
weather_map   = {"Clear":0,"Cloudy":1,"Rainy":2,"Foggy":3}
road_map      = {"Good":0,"Wet":1,"Damaged":2,"Under repair":3}
light_map     = {"Day":0,"Night":1}
road_type_map = {"Urban":0,"Rural":1,"Highway":2,"Intersection":3}

df["weather_encoded"]   = df["weather"].map(weather_map)
df["road_cond_encoded"] = df["road_condition"].map(road_map)
df["light_encoded"]     = df["light_condition"].map(light_map)
df["road_type_encoded"] = df["road_type"].map(road_type_map)

df.drop(columns=["date","time"], inplace=True)
df.drop_duplicates(inplace=True)

df.to_csv(f"{PROCESSED}/accident_clean.csv", index=False)
print(f"  Saved → data/processed/accident_clean.csv  ({len(df):,} rows)")


# ════════════════════════════════════════════════
# 3. CRIME DATA
# ════════════════════════════════════════════════
print("\n[3/3] Cleaning crime data...")
df = pd.read_csv(f"{RAW}/crime_data.csv")
report(df, "CRIME — raw")

df["datetime"]     = pd.to_datetime(df["date"] + " " + df["time"])
df["is_rush_hour"] = df["hour"].apply(lambda h: 1 if (7<=h<=9 or 17<=h<=19) else 0)

# Encode crime type
crime_types = sorted(df["crime_type"].unique())
crime_map   = {c:i for i,c in enumerate(crime_types)}
df["crime_type_encoded"] = df["crime_type"].map(crime_map)

# Binary target: high risk if risk_score > 0.5
df["high_risk"] = (df["risk_score"] > 0.5).astype(int)

df.drop(columns=["date","time"], inplace=True)
df.drop_duplicates(inplace=True)

df.to_csv(f"{PROCESSED}/crime_clean.csv", index=False)
print(f"  Saved → data/processed/crime_clean.csv  ({len(df):,} rows)")


# ════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════
print("\n" + "="*50)
print("Data cleaning complete!")
print("Files in data/processed/:")
for f in os.listdir(PROCESSED):
    path = f"{PROCESSED}/{f}"
    rows = len(pd.read_csv(path))
    cols = len(pd.read_csv(path).columns)
    print(f"  {f:30s}  {rows:,} rows  x  {cols} columns")
print("="*50)
print("\nNext step: Train ML models!")