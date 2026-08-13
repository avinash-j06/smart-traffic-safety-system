# 🚦 Smart Traffic, Accident & Crime Prediction + Safe Route System

<p align="center">
  <img src="https://img.shields.io/badge/AI%20Powered-Smart%20Safety%20System-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Machine%20Learning-3%20Models-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Location-Raipur%2C%20India-orange?style=for-the-badge" />
</p>

<p align="center">
  <a href="https://smart-traffic-raipur.onrender.com">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Open%20Application-success?style=for-the-badge" />
  </a>
  <a href="https://github.com/avinash-j06/smart-traffic-safety-system">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github" />
  </a>
</p>

<p align="center">
  <b>An AI-powered traffic and public safety intelligence system for Raipur, Chhattisgarh.</b>
</p>

---

# 🌐 Live Demo

<p align="center">

### 🚀 [Open Smart Traffic Safety System](https://smart-traffic-raipur.onrender.com)

</p>

> ⚠️ **Note:** The application is hosted on Render. If the server is inactive, the first request may take a few moments to start.

---

# 📌 Project Overview

Smart Traffic Safety System is a **full-stack Machine Learning web application** designed to analyze traffic conditions, accident risks, and crime hotspots across different locations in **Raipur, Chhattisgarh**.

The system combines predictions from **three Machine Learning models** and generates a unified **Safety Score (0–100)** to help users understand how safe a location is for travel.

It also includes an interactive **Safe Route Planner** that recommends safer travel routes between locations.

## 🧠 What the System Can Do?

```text
                📍 Select Location
                        │
                        ▼
        ┌───────────────────────────────┐
        │     SMART SAFETY ENGINE       │
        └───────────────────────────────┘
                 │       │       │
                 ▼       ▼       ▼
             🚦 Traffic ⚠️ Accident 🔴 Crime
                 │       │       │
                 └───────┼───────┘
                         ▼
                  🧮 Safety Score
                         │
                         ▼
              🟢 Safe / ⚠️ Caution / 🔴 Avoid
                         │
                         ▼
                  🗺️ Safe Route
```

---

# ✨ Key Features

| Feature                         | Description                                                               |
| ------------------------------- | ------------------------------------------------------------------------- |
| 🚦 **Traffic Prediction**       | Predicts Free, Moderate, or Heavy traffic congestion                      |
| ⚠️ **Accident Risk Analysis**   | Predicts accident severity based on road and environmental conditions     |
| 🔴 **Crime Hotspot Prediction** | Identifies High or Low crime-risk areas                                   |
| 🧮 **Safety Score**             | Combines traffic, accident, and crime predictions into a score from 0–100 |
| 🗺️ **Safe Route Planner**      | Recommends safer routes between Raipur locations                          |
| 📍 **Interactive Map**          | Visualizes locations, routes, and risk zones                              |
| 🔥 **Heatmaps**                 | Displays Traffic, Crime, and Accident hotspots                            |
| 📊 **Live Analytics**           | Interactive charts and hourly risk visualization                          |
| 🕒 **Real-Time Clock**          | Displays current system time on the dashboard                             |

---

# 🖥️ Dashboard Preview

```text
┌──────────────────────────────────────────────────────────────┐
│ 🚦 SMART TRAFFIC & SAFETY SYSTEM                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 📍 Location: Pandri                                          │
│                                                              │
│ 🚦 Traffic: HEAVY                                            │
│ ⚠️ Accident Risk: MODERATE                                  │
│ 🔴 Crime Risk: HIGH                                         │
│                                                              │
│ 🧮 Safety Score: 35 / 100                                   │
│                                                              │
│ 🔴 Recommendation: AVOID / USE SAFER ROUTE                  │
│                                                              │
│ 🗺️ Interactive Map + Heatmaps + Safe Route                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

# 🛠️ Technology Stack

## 🧠 Machine Learning

| Technology            | Usage                            |
| --------------------- | -------------------------------- |
| **XGBoost**           | Traffic Congestion Prediction    |
| **Random Forest**     | Accident Severity Prediction     |
| **Gradient Boosting** | Crime Risk Prediction            |
| **Scikit-learn**      | ML model training and evaluation |
| **Joblib**            | Model serialization              |

## ⚙️ Backend

* Python 3.13
* Flask
* Flask-CORS
* REST API

## 🎨 Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js

## 🗺️ Maps & Visualization

* Leaflet.js
* OpenStreetMap
* Leaflet.heat

## 📊 Data Processing

* Pandas
* NumPy

---

# 🤖 Machine Learning Models

## 1️⃣ Traffic Congestion Predictor 🚦

| Property   | Details                                        |
| ---------- | ---------------------------------------------- |
| Algorithm  | XGBoost Classifier                             |
| Prediction | Free / Moderate / Heavy                        |
| Accuracy   | **99.8%**                                      |
| Features   | Hour, Vehicle Count, Speed, Weather, Rush Hour |

### Example

```text
Input:
Hour = 18
Weather = Rainy
Vehicle Count = 700
Speed = 20 km/h

↓

Prediction

🚦 HEAVY TRAFFIC
```

---

## 2️⃣ Accident Severity Predictor ⚠️

| Property   | Details                                         |
| ---------- | ----------------------------------------------- |
| Algorithm  | Random Forest Classifier                        |
| Prediction | Minor / Moderate / Severe / Fatal               |
| Features   | Road Type, Weather, Time, Speed Limit, Vehicles |

### Example

```text
Rainy Weather
      +
High Vehicle Speed
      +
Busy Road
      ↓

⚠️ SEVERE ACCIDENT RISK
```

---

## 3️⃣ Crime Risk Predictor 🔴

| Property   | Details                                    |
| ---------- | ------------------------------------------ |
| Algorithm  | Gradient Boosting Classifier               |
| Prediction | High Risk / Low Risk                       |
| Accuracy   | **85.8%**                                  |
| Features   | Hour, Night Flag, Day, Crime Type, Weekend |

### Example

```text
Late Night
    +
Weekend
    +
High Risk Zone
        ↓

🔴 HIGH CRIME RISK
```

---

# 🧮 Safety Score System

The system combines predictions from all three models.

```text
                🚦 Traffic Risk
                      │
                      ▼
                    -15
                      │
                      ▼
⚠️ Accident Risk ──→ SAFETY SCORE ←── 🔴 Crime Risk
                      │
                      ▼
                     0–100
```

### Formula

```text
Safety Score =
100
- (Traffic Level × 15)
- (Accident Severity × 10)
- (Crime Probability × 30)
```

### Score Interpretation

| Score           | Status  | Recommendation       |
| --------------- | ------- | -------------------- |
| 🟢 **70–100**   | Safe    | Safe to Travel       |
| 🟡 **50–69**    | Caution | Travel Carefully     |
| 🔴 **Below 50** | Avoid   | Consider Safer Route |

---

# 📡 API Endpoints

| Method | Endpoint                | Description                        |
| ------ | ----------------------- | ---------------------------------- |
| `GET`  | `/`                     | API health check                   |
| `GET`  | `/dashboard`            | Interactive dashboard              |
| `POST` | `/api/predict/traffic`  | Predict traffic congestion         |
| `POST` | `/api/predict/accident` | Predict accident severity          |
| `POST` | `/api/predict/crime`    | Predict crime risk                 |
| `POST` | `/api/predict/all`      | Combined prediction + Safety Score |
| `POST` | `/api/route/safe`       | Find safest route                  |

---

# 🔄 Example API Request

### Endpoint

```text
POST /api/predict/all
```

### Request

```json
{
  "location": "Pandri",
  "hour": 20,
  "weather": "Rainy",
  "vehicle_count": 600,
  "avg_speed_kmph": 25
}
```

### Response

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

# 🗺️ Safe Route Planning

The Safe Route Planner analyzes multiple locations and recommends a route with a better overall safety profile.

```text
📍 Start Location
      │
      ▼
🚦 Analyze Traffic
      │
      ▼
⚠️ Analyze Accident Risk
      │
      ▼
🔴 Analyze Crime Risk
      │
      ▼
🧮 Calculate Safety Score
      │
      ▼
🗺️ Recommend Safer Route
      │
      ▼
🏁 Destination
```

---

# 📍 Monitored Locations

The project analyzes multiple areas across **Raipur, Chhattisgarh**.

```text
📍 Pandri
📍 Shankar Nagar
📍 Telibandha
📍 Tikrapara
📍 Amanaka
📍 Fafadih
📍 Devendra Nagar
📍 Mowa
📍 Katora Talab
📍 GE Road
📍 Jail Road
📍 Ring Road No. 1
```

---

# 📊 Dataset Information

The datasets are synthetically generated using realistic patterns and location data for Raipur.

| Dataset     | Records | Description                           |
| ----------- | ------: | ------------------------------------- |
| 🚦 Traffic  |   5,000 | Traffic conditions and congestion     |
| ⚠️ Accident |   1,500 | Accident severity and road conditions |
| 🔴 Crime    |   2,000 | Crime patterns and risk levels        |

---

# 📁 Project Structure

```text
smart-traffic-safety-system/
│
├── data/
│   ├── raw/
│   │   ├── traffic_data.csv
│   │   ├── accident_data.csv
│   │   └── crime_data.csv
│   │
│   └── processed/
│       ├── traffic_clean.csv
│       ├── accident_clean.csv
│       └── crime_clean.csv
│
├── models/
│   ├── traffic_model.pkl
│   ├── accident_model.pkl
│   └── crime_model.pkl
│
├── notebooks/
│   ├── generate_data.py
│   ├── clean_data.py
│   └── train_models.py
│
├── backend/
│   ├── app.py
│   └── config.py
│
├── frontend/
│   ├── templates/
│   │   └── index.html
│   │
│   └── static/
│
├── requirements.txt
│
└── README.md
```

---

# 🚀 Run Locally

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/avinash-j06/smart-traffic-safety-system.git
```

## 2️⃣ Open the Project

```bash
cd smart-traffic-safety-system
```

## 3️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 5️⃣ Generate the Dataset

```bash
python notebooks/generate_data.py
```

## 6️⃣ Clean and Process Data

```bash
python notebooks/clean_data.py
```

## 7️⃣ Train the Models

```bash
python notebooks/train_models.py
```

## 8️⃣ Start the Application

```bash
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000/dashboard
```

---

# 🌐 Deployment

The project is deployed online using **Render**.

### 🚀 Live Application

[🔗 Open Smart Traffic Safety System](https://smart-traffic-raipur.onrender.com)

---

# 🔮 Future Improvements

* 🤖 AI chatbot for travel safety assistance
* 🎙️ Voice-based navigation
* 📡 Integration with real-time traffic APIs
* 🚨 Emergency alert system
* 📱 Mobile application
* 🛰️ IoT-based traffic monitoring
* 📈 Continuous ML model retraining
* 🚗 Personalized safe route recommendations
* 🧠 Advanced Deep Learning models

---

# 👨‍💻 Project Team

| Name                | Role                           |
| ------------------- | ------------------------------ |
| **Avinash Jaiswal** | Development & Machine Learning |
| **Birju Ram Sahu**  | Development & Research         |

### 🏫 Academic Details

* **College:** Government Engineering College, Raipur
* **Branch:** Computer Science & Engineering
* **Project Guide:** Mrs. Anjum Khan

---

# 🙏 Acknowledgements

Special thanks to the open-source technologies and resources that supported this project:

* 🗺️ OpenStreetMap contributors
* 🗺️ Leaflet.js
* 📊 Chart.js
* 🧠 Scikit-learn
* ⚡ XGBoost
* 📚 NCRB for real-world crime data references

---

# 📄 License

This project is developed for **academic and educational purposes**.

---

<p align="center">

### ⭐ If you found this project interesting, consider giving it a star!

**Made with ❤️ using Machine Learning, Python, and Web Technologies**

[🚀 Try Live Demo](https://smart-traffic-raipur.onrender.com) • [💻 View Source Code](https://github.com/avinash-j06/smart-traffic-safety-system)

</p>
