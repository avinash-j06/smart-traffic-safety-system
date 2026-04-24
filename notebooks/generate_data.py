import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

# So we get same data every time we run
np.random.seed(42)
random.seed(42)

# Output folder
OUTPUT = "data/raw"
os.makedirs(OUTPUT, exist_ok=True)

# ── Raipur locations ──────────────────────────────────────────
LOCATIONS = [
    {"name": "Pandri",          "lat": 21.2514, "lon": 81.6296},
    {"name": "Shankar Nagar",   "lat": 21.2435, "lon": 81.6421},
    {"name": "Telibandha",      "lat": 21.2389, "lon": 81.6501},
    {"name": "Tikrapara",       "lat": 21.2601, "lon": 81.6187},
    {"name": "Amanaka",         "lat": 21.2700, "lon": 81.6350},
    {"name": "Fafadih",         "lat": 21.2331, "lon": 81.6612},
    {"name": "Devendra Nagar",  "lat": 21.2456, "lon": 81.6234},
    {"name": "Mowa",            "lat": 21.2789, "lon": 81.6423},
    {"name": "Katora Talab",    "lat": 21.2367, "lon": 81.6345},
    {"name": "GE Road",         "lat": 21.2512, "lon": 81.6489},
    {"name": "Jail Road",       "lat": 21.2298, "lon": 81.6401},
    {"name": "Ring Road No 1",  "lat": 21.2634, "lon": 81.6512},
]

print("Generating datasets for Raipur, Chhattisgarh...")
print("=" * 50)


# ════════════════════════════════════════════════
# 1. TRAFFIC DATA  (5 000 records)
# ════════════════════════════════════════════════
print("\n[1/3] Generating traffic data...")

records = []
start_date = datetime(2023, 1, 1)

for i in range(5000):
    loc   = random.choice(LOCATIONS)
    dt    = start_date + timedelta(
                days=random.randint(0, 364),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59))
    hour  = dt.hour
    dow   = dt.weekday()          # 0=Mon … 6=Sun

    # Rush hours → more congestion
    is_rush = (7 <= hour <= 9) or (17 <= hour <= 19)
    is_weekend = dow >= 5

    base_volume = random.randint(200, 800)
    if is_rush:    base_volume = int(base_volume * 1.8)
    if is_weekend: base_volume = int(base_volume * 0.7)

    speed = random.uniform(15, 60)
    if is_rush: speed = random.uniform(5, 25)

    # Congestion level  0=free  1=moderate  2=heavy
    if speed < 15:        congestion = 2
    elif speed < 30:      congestion = 1
    else:                 congestion = 0

    # Weather effect
    weather = random.choices(
        ["Clear","Rainy","Foggy","Cloudy"],
        weights=[60, 20, 10, 10])[0]
    if weather == "Rainy":  speed  *= 0.7
    if weather == "Foggy":  speed  *= 0.6

    records.append({
        "date":            dt.strftime("%Y-%m-%d"),
        "time":            dt.strftime("%H:%M"),
        "hour":            hour,
        "day_of_week":     dow,
        "location":        loc["name"],
        "latitude":        loc["lat"] + np.random.normal(0, 0.002),
        "longitude":       loc["lon"] + np.random.normal(0, 0.002),
        "vehicle_count":   base_volume,
        "avg_speed_kmph":  round(max(speed, 2), 1),
        "weather":         weather,
        "is_holiday":      1 if random.random() < 0.05 else 0,
        "congestion_level":congestion,
    })

traffic_df = pd.DataFrame(records)
traffic_df.to_csv(f"{OUTPUT}/traffic_data.csv", index=False)
print(f"   Saved {len(traffic_df)} records → data/raw/traffic_data.csv")


# ════════════════════════════════════════════════
# 2. ACCIDENT DATA  (1 500 records)
# ════════════════════════════════════════════════
print("\n[2/3] Generating accident data...")

ACCIDENT_TYPES  = ["Collision","Pedestrian","Skid","Overturning","Hit-and-run"]
SEVERITY_LABELS = ["Minor","Moderate","Severe","Fatal"]
ROAD_TYPES      = ["Highway","Urban","Rural","Intersection"]

records = []
for i in range(1500):
    loc  = random.choice(LOCATIONS)
    dt   = start_date + timedelta(
               days=random.randint(0, 364),
               hours=random.randint(0, 23))
    hour = dt.hour
    dow  = dt.weekday()

    # Night + rain → higher severity
    weather  = random.choices(
        ["Clear","Rainy","Foggy","Cloudy"],
        weights=[55, 25, 12, 8])[0]
    is_night = hour < 6 or hour > 21

    sev_weights = [40, 30, 20, 10]
    if is_night:          sev_weights = [20, 30, 30, 20]
    if weather == "Rainy":sev_weights = [25, 30, 30, 15]

    severity_idx = random.choices([0,1,2,3], weights=sev_weights)[0]

    records.append({
        "date":             dt.strftime("%Y-%m-%d"),
        "time":             dt.strftime("%H:%M"),
        "hour":             hour,
        "day_of_week":      dow,
        "location":         loc["name"],
        "latitude":         loc["lat"] + np.random.normal(0, 0.003),
        "longitude":        loc["lon"] + np.random.normal(0, 0.003),
        "accident_type":    random.choice(ACCIDENT_TYPES),
        "severity":         SEVERITY_LABELS[severity_idx],
        "severity_encoded": severity_idx,
        "vehicles_involved":random.randint(1, 4),
        "casualties":       random.randint(0, severity_idx + 1),
        "road_type":        random.choice(ROAD_TYPES),
        "weather":          weather,
        "light_condition":  "Night" if is_night else "Day",
        "speed_limit":      random.choice([40, 60, 80, 100]),
        "road_condition":   random.choices(
                                ["Good","Wet","Damaged","Under repair"],
                                weights=[50,25,15,10])[0],
    })

accident_df = pd.DataFrame(records)
accident_df.to_csv(f"{OUTPUT}/accident_data.csv", index=False)
print(f"   Saved {len(accident_df)} records → data/raw/accident_data.csv")


# ════════════════════════════════════════════════
# 3. CRIME DATA  (2 000 records)
# ════════════════════════════════════════════════
print("\n[3/3] Generating crime data...")

CRIME_TYPES = [
    "Theft","Robbery","Assault","Vehicle theft",
    "Burglary","Snatching","Harassment","Vandalism"
]
STATUS_OPTS = ["Reported","Under investigation","Solved","Closed"]

records = []
for i in range(2000):
    loc  = random.choice(LOCATIONS)
    dt   = start_date + timedelta(
               days=random.randint(0, 364),
               hours=random.randint(0, 23))
    hour = dt.hour
    dow  = dt.weekday()

    is_night = hour < 6 or hour > 20

    # Night crimes skew toward theft / robbery
    if is_night:
        crime = random.choices(
            CRIME_TYPES,
            weights=[25,20,15,15,10,10,3,2])[0]
        risk = random.uniform(0.5, 1.0)
    else:
        crime = random.choices(
            CRIME_TYPES,
            weights=[20,10,10,15,10,15,12,8])[0]
        risk = random.uniform(0.1, 0.6)

    records.append({
        "date":           dt.strftime("%Y-%m-%d"),
        "time":           dt.strftime("%H:%M"),
        "hour":           hour,
        "day_of_week":    dow,
        "month":          dt.month,
        "location":       loc["name"],
        "latitude":       loc["lat"] + np.random.normal(0, 0.004),
        "longitude":      loc["lon"] + np.random.normal(0, 0.004),
        "crime_type":     crime,
        "is_night":       int(is_night),
        "is_weekend":     int(dow >= 5),
        "risk_score":     round(risk, 3),
        "status":         random.choices(
                              STATUS_OPTS,
                              weights=[30,40,20,10])[0],
    })

crime_df = pd.DataFrame(records)
crime_df.to_csv(f"{OUTPUT}/crime_data.csv", index=False)
print(f"   Saved {len(crime_df)} records → data/raw/crime_data.csv")


# ════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════
print("\n" + "=" * 50)
print("All datasets generated successfully!")
print(f"  Traffic  : {len(traffic_df):,} rows  x  {len(traffic_df.columns)} columns")
print(f"  Accidents: {len(accident_df):,} rows  x  {len(accident_df.columns)} columns")
print(f"  Crime    : {len(crime_df):,} rows  x  {len(crime_df.columns)} columns")
print("\nFirst 3 rows of traffic data:")
print(traffic_df.head(3).to_string())