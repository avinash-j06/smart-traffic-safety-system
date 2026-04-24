# 🚦 Smart Traffic, Accident & Crime Prediction + Safe Route System

> A machine learning powered web application for real-time traffic monitoring,
> accident risk assessment, crime hotspot prediction, and safe route planning
> for **Raipur, Chhattisgarh, India**.

---

## 📌 Project Overview

This project is a full-stack intelligent traffic and safety system that uses
machine learning models to predict traffic congestion, accident severity, and
crime risk for different locations in Raipur. It also recommends the safest
route between two locations based on live ML predictions.

The system consists of:
- A **Python data pipeline** for data generation and cleaning
- Three **ML models** trained on realistic Raipur area data
- A **Flask REST API** serving predictions via HTTP endpoints
- An **interactive web dashboard** with a live map, heatmaps, and charts

---

## 🎯 Features

| Feature | Description |
|---|---|
| 🚦 Traffic Congestion Prediction | Predicts Free / Moderate / Heavy congestion based on hour, weather, vehicle count |
| ⚠️ Accident Severity Prediction | Predicts Minor / Moderate / Severe / Fatal risk based on road and weather conditions |
| 🔴 Crime Risk Prediction | Predicts High / Low crime risk for a location and time |
| 📍 Combined Location Analysis | Single API call gives traffic + accident + crime + safety score for any location |
| 🗺️ Safe Route Planner | Recommends the safest path between two Raipur locations |
| 📊 Interactive Dashboard | Dark-themed web UI with live Leaflet.js map, heatmaps, and Chart.js visualizations |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| ML Models | XGBoost, Random Forest, Gradient Boosting (scikit-learn) |
| Backend | Flask, Flask-CORS |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Frontend | HTML5, CSS3, JavaScript |
| Maps | Leaflet.js + OpenStreetMap (free, open-source) |
| Charts | Chart.js |
| Heatmaps | Leaflet.heat plugin |

---

## 📁 Project Structure

```
smart traffic project/
│
├── data/
│   ├── raw/                        # Original generated datasets
│   │   ├── traffic_data.csv        # 5,000 traffic records
│   │   ├── accident_data.csv       # 1,500 accident records
│   │   └── crime_data.csv          # 2,000 crime records
│   └── processed/                  # Cleaned & feature-engineered data
│       ├── traffic_clean.csv
│       ├── accident_clean.csv
│       └── crime_clean.csv
│
├── models/                         # Trained ML model files
│   ├── traffic_model.pkl           # XGBoost — 99.8% accuracy
│   ├── accident_model.pkl          # Random Forest
│   └── crime_model.pkl             # Gradient Boosting — 85.8% accuracy
│
├── notebooks/                      # Data pipeline scripts
│   ├── generate_data.py            # Generates synthetic Raipur datasets
│   ├── clean_data.py               # Cleans and engineers features
│   └── train_models.py             # Trains and saves all 3 ML models
│
├── backend/
│   ├── app.py                      # Flask API with all endpoints
│   └── config.py                   # Configuration settings
│
├── frontend/
│   ├── templates/
│   │   └── index.html              # Main dashboard page
│   └── static/                     # CSS, JS, images
│
├── .env                            # Environment variables
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🤖 ML Models

### Model 1 — Traffic Congestion Predictor
- **Algorithm:** XGBoost Classifier
- **Target:** Congestion level (0=Free, 1=Moderate, 2=Heavy)
- **Accuracy:** 99.8%
- **Key features:** Hour, vehicle count, speed, weather, rush hour flag

### Model 2 — Accident Severity Predictor
- **Algorithm:** Random Forest Classifier
- **Target:** Severity (0=Minor, 1=Moderate, 2=Severe, 3=Fatal)
- **Key features:** Road type, weather, time of day, speed limit, vehicles involved

### Model 3 — Crime Risk Predictor
- **Algorithm:** Gradient Boosting Classifier
- **Target:** High risk (1) or Low risk (0)
- **Accuracy:** 85.8%
- **Key features:** Hour, is_night, day of week, crime type, weekend flag

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check — lists all endpoints |
| GET | `/dashboard` | Serves the web dashboard |
| POST | `/api/predict/traffic` | Predict traffic congestion |
| POST | `/api/predict/accident` | Predict accident severity |
| POST | `/api/predict/crime` | Predict crime risk |
| POST | `/api/predict/all` | Combined prediction + safety score |
| POST | `/api/route/safe` | Find safest route between two locations |

### Example API Request

**POST** `/api/predict/all`

```json
{
  "location": "Pandri",
  "hour": 20,
  "weather": "Rainy",
  "vehicle_count": 600,
  "avg_speed_kmph": 25
}
```

**Response:**
```json
{
  "location": "Pandri",
  "traffic": "Heavy",
  "accident_risk": "Moderate",
  "crime_risk": "High",
  "safety_score": 35,
  "safe_to_travel": false
}
```

---

## 🚀 How to Run

### 1. Clone / open the project folder
```bash
cd "E:\smart traffic project"
```

### 2. Activate virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate datasets (first time only)
```bash
python notebooks/generate_data.py
```

### 5. Clean data (first time only)
```bash
python notebooks/clean_data.py
```

### 6. Train models (first time only)
```bash
python notebooks/train_models.py
```

### 7. Start the Flask API
```bash
python backend/app.py
```

### 8. Open the dashboard
Open your browser and go to:
```
http://127.0.0.1:5000/dashboard
```

---

## 📊 Dataset Description

All datasets are synthetically generated with realistic patterns for
**Raipur, Chhattisgarh** using actual area coordinates.

### Monitored Locations (12 zones)
Pandri · Shankar Nagar · Telibandha · Tikrapara · Amanaka · Fafadih ·
Devendra Nagar · Mowa · Katora Talab · GE Road · Jail Road · Ring Road No 1

### Traffic Dataset (5,000 records)
- Columns: datetime, location, lat/lon, vehicle count, speed, weather,
  congestion level, rush hour flags

### Accident Dataset (1,500 records)
- Columns: datetime, location, accident type, severity, road type,
  weather, light condition, speed limit, casualties

### Crime Dataset (2,000 records)
- Columns: datetime, location, crime type, risk score, is_night,
  is_weekend, status

---

## 🔮 Safety Score Formula

The overall safety score (0–100) for a location is calculated as:

```
Safety Score = 100 - (traffic_level × 15) - (accident_severity × 10) - (crime_probability × 30)
```

- Score ≥ 70 → ✅ Safe
- Score 50–69 → ⚠️ Caution
- Score < 50 → 🔴 Avoid

---

## 📱 Dashboard Features

- **Live Map** — OpenStreetMap with Raipur area markers
- **Heatmap Layers** — Toggle between Traffic / Crime / Accident heatmaps
- **Location Predictor** — Select any zone and get instant ML predictions
- **Safe Route Planner** — Visual route drawn on map with colour-coded stops
- **Hourly Risk Chart** — 24-hour traffic and crime risk profile
- **Real-time Clock** — Live time display

---

## 👨‍💻 Author

**Student Name:** *Avinash Jaiswal*
**Student Name:** *Birju Ram Sahu*
**College:** *GEC Raipur*
**Branch:** *(Computer Science Engineering)*
**Year:** *(Pre-Final Year — 2025–26)*
**Guide:** *Mrs.Anjum Khan*

---

## 📝 Acknowledgements

- OpenStreetMap contributors for free map tiles
- Leaflet.js for the open-source mapping library
- NCRB (National Crime Records Bureau) for real-world data reference
- scikit-learn, XGBoost teams for ML libraries

---

## 📄 License

This project is built for academic purposes.
