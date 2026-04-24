from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__,
            template_folder="../frontend/templates",
            static_folder="../frontend/static")
CORS(app)

# ── Load all 3 models at startup ──────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "..", "models")

print("Loading models...")
traffic_bundle  = joblib.load(os.path.join(MODELS, "traffic_model.pkl"))
accident_bundle = joblib.load(os.path.join(MODELS, "accident_model.pkl"))
crime_bundle    = joblib.load(os.path.join(MODELS, "crime_model.pkl"))

traffic_model   = traffic_bundle["model"]
accident_model  = accident_bundle["model"]
crime_model     = crime_bundle["model"]
print("All models loaded!")

# ── Encoding maps (must match clean_data.py) ──────────────────
WEATHER_MAP   = {"Clear":0,"Cloudy":1,"Rainy":2,"Foggy":3}
ROAD_COND_MAP = {"Good":0,"Wet":1,"Damaged":2,"Under repair":3}
LIGHT_MAP     = {"Day":0,"Night":1}
ROAD_TYPE_MAP = {"Urban":0,"Rural":1,"Highway":2,"Intersection":3}
CRIME_TYPE_MAP = {
    "Assault":0,"Burglary":1,"Harassment":2,
    "Robbery":3,"Snatching":4,"Theft":5,
    "Vandalism":6,"Vehicle theft":7
}

CONGESTION_LABELS = {0:"Free flow",1:"Moderate",2:"Heavy"}
SEVERITY_LABELS   = {0:"Minor",1:"Moderate",2:"Severe",3:"Fatal"}

# ── Helper ────────────────────────────────────────────────────
def get_time_features(hour=None, day_of_week=None, month=None):
    now = datetime.now()
    hour        = hour        if hour        is not None else now.hour
    day_of_week = day_of_week if day_of_week is not None else now.weekday()
    month       = month       if month       is not None else now.month
    is_weekend   = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if (7<=hour<=9 or 17<=hour<=19) else 0
    is_night     = 1 if (hour<6 or hour>21) else 0
    return hour, day_of_week, month, is_weekend, is_rush_hour, is_night


# ════════════════════════════════════════════════
# ROUTE 1 — Health check
# ════════════════════════════════════════════════
@app.route("/")
def home():
    return jsonify({
        "status" : "running",
        "message": "Smart Traffic API is live!",
        "endpoints": [
            "/api/predict/traffic",
            "/api/predict/accident",
            "/api/predict/crime",
            "/api/predict/all",
            "/api/route/safe"
        ]
    })


# ════════════════════════════════════════════════
# ROUTE 2 — Traffic congestion prediction
# POST /api/predict/traffic
# Body: { "hour":8, "vehicle_count":600,
#         "avg_speed_kmph":20, "weather":"Rainy",
#         "is_holiday":0 }
# ════════════════════════════════════════════════
@app.route("/api/predict/traffic", methods=["POST"])
def predict_traffic():
    try:
        data = request.get_json()

        hour, dow, month, is_weekend, is_rush, _ = get_time_features(
            data.get("hour"), data.get("day_of_week"), data.get("month"))

        weather_enc = WEATHER_MAP.get(data.get("weather","Clear"), 0)

        features = [[
            hour, dow, month, is_weekend, is_rush,
            int(data.get("vehicle_count", 400)),
            float(data.get("avg_speed_kmph", 30)),
            weather_enc,
            int(data.get("is_holiday", 0))
        ]]

        pred  = int(traffic_model.predict(features)[0])
        proba = traffic_model.predict_proba(features)[0].tolist()

        return jsonify({
            "prediction"      : pred,
            "label"           : CONGESTION_LABELS[pred],
            "confidence"      : round(max(proba)*100, 1),
            "probabilities"   : {
                "Free flow" : round(proba[0]*100,1),
                "Moderate"  : round(proba[1]*100,1),
                "Heavy"     : round(proba[2]*100,1)
            },
            "input_used": {
                "hour":hour, "day_of_week":dow,
                "weather":data.get("weather","Clear"),
                "vehicle_count":data.get("vehicle_count",400)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ════════════════════════════════════════════════
# ROUTE 3 — Accident severity prediction
# POST /api/predict/accident
# Body: { "hour":22, "weather":"Rainy",
#         "road_condition":"Wet", "road_type":"Highway",
#         "vehicles_involved":2, "speed_limit":80 }
# ════════════════════════════════════════════════
@app.route("/api/predict/accident", methods=["POST"])
def predict_accident():
    try:
        data = request.get_json()

        hour, dow, month, is_weekend, is_rush, is_night = get_time_features(
            data.get("hour"), data.get("day_of_week"), data.get("month"))

        features = [[
            hour, dow, month, is_weekend, is_night, is_rush,
            int(data.get("vehicles_involved", 1)),
            int(data.get("speed_limit", 60)),
            WEATHER_MAP.get(data.get("weather","Clear"), 0),
            ROAD_COND_MAP.get(data.get("road_condition","Good"), 0),
            LIGHT_MAP.get("Night" if is_night else "Day", 0),
            ROAD_TYPE_MAP.get(data.get("road_type","Urban"), 0)
        ]]

        pred  = int(accident_model.predict(features)[0])
        proba = accident_model.predict_proba(features)[0].tolist()

        return jsonify({
            "prediction"   : pred,
            "label"        : SEVERITY_LABELS[pred],
            "confidence"   : round(max(proba)*100, 1),
            "probabilities": {
                "Minor"    : round(proba[0]*100,1),
                "Moderate" : round(proba[1]*100,1),
                "Severe"   : round(proba[2]*100,1),
                "Fatal"    : round(proba[3]*100,1)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ════════════════════════════════════════════════
# ROUTE 4 — Crime risk prediction
# POST /api/predict/crime
# Body: { "hour":23, "crime_type":"Theft" }
# ════════════════════════════════════════════════
@app.route("/api/predict/crime", methods=["POST"])
def predict_crime():
    try:
        data = request.get_json()

        hour, dow, month, is_weekend, is_rush, is_night = get_time_features(
            data.get("hour"), data.get("day_of_week"), data.get("month"))

        crime_enc = CRIME_TYPE_MAP.get(data.get("crime_type","Theft"), 5)

        features = [[
            hour, dow, month, is_night,
            is_weekend, is_rush, crime_enc
        ]]

        pred  = int(crime_model.predict(features)[0])
        proba = crime_model.predict_proba(features)[0].tolist()

        return jsonify({
            "prediction" : pred,
            "label"      : "High Risk" if pred==1 else "Low Risk",
            "risk_score" : round(proba[1]*100, 1),
            "confidence" : round(max(proba)*100, 1),
            "is_night"   : bool(is_night),
            "is_weekend" : bool(is_weekend)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ════════════════════════════════════════════════
# ROUTE 5 — Combined prediction for a location
# POST /api/predict/all
# Body: { "hour":20, "weather":"Rainy",
#         "location":"Pandri" }
# ════════════════════════════════════════════════
@app.route("/api/predict/all", methods=["POST"])
def predict_all():
    try:
        data = request.get_json()
        hour, dow, month, is_weekend, is_rush, is_night = get_time_features(
            data.get("hour"), data.get("day_of_week"), data.get("month"))

        weather_enc = WEATHER_MAP.get(data.get("weather","Clear"), 0)

        # Traffic
        t_feat = [[hour, dow, month, is_weekend, is_rush,
                   int(data.get("vehicle_count",400)),
                   float(data.get("avg_speed_kmph",30)),
                   weather_enc, int(data.get("is_holiday",0))]]
        t_pred = int(traffic_model.predict(t_feat)[0])

        # Accident
        a_feat = [[hour, dow, month, is_weekend, is_night, is_rush,
                   int(data.get("vehicles_involved",1)),
                   int(data.get("speed_limit",60)),
                   weather_enc,
                   ROAD_COND_MAP.get(data.get("road_condition","Good"),0),
                   LIGHT_MAP.get("Night" if is_night else "Day",0),
                   ROAD_TYPE_MAP.get(data.get("road_type","Urban"),0)]]
        a_pred = int(accident_model.predict(a_feat)[0])

        # Crime
        c_feat = [[hour, dow, month, is_night, is_weekend,
                   is_rush, CRIME_TYPE_MAP.get(data.get("crime_type","Theft"),5)]]
        c_pred = int(crime_model.predict(c_feat)[0])
        c_prob = crime_model.predict_proba(c_feat)[0][1]

        # Overall safety score 0-100 (higher = safer)
        traffic_penalty = t_pred * 15
        accident_penalty = a_pred * 10
        crime_penalty    = c_prob * 30
        safety_score = max(0, round(100 - traffic_penalty - accident_penalty - crime_penalty))

        return jsonify({
            "location"      : data.get("location","Unknown"),
            "hour"          : hour,
            "weather"       : data.get("weather","Clear"),
            "traffic"       : CONGESTION_LABELS[t_pred],
            "accident_risk" : SEVERITY_LABELS[a_pred],
            "crime_risk"    : "High" if c_pred==1 else "Low",
            "safety_score"  : safety_score,
            "safe_to_travel": safety_score >= 50
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ════════════════════════════════════════════════
# ROUTE 6 — Safe route between two locations
# POST /api/route/safe
# Body: { "origin":"Pandri",
#         "destination":"Mowa",
#         "hour":21, "weather":"Clear" }
# ════════════════════════════════════════════════
LOCATION_COORDS = {
    "Pandri"         : (21.2514, 81.6296),
    "Shankar Nagar"  : (21.2435, 81.6421),
    "Telibandha"     : (21.2389, 81.6501),
    "Tikrapara"      : (21.2601, 81.6187),
    "Amanaka"        : (21.2700, 81.6350),
    "Fafadih"        : (21.2331, 81.6612),
    "Devendra Nagar" : (21.2456, 81.6234),
    "Mowa"           : (21.2789, 81.6423),
    "Katora Talab"   : (21.2367, 81.6345),
    "GE Road"        : (21.2512, 81.6489),
    "Jail Road"      : (21.2298, 81.6401),
    "Ring Road No 1" : (21.2634, 81.6512),
}

@app.route("/api/route/safe", methods=["POST"])
def safe_route():
    try:
        data   = request.get_json()
        origin = data.get("origin","Pandri")
        dest   = data.get("destination","Mowa")
        hour   = data.get("hour", datetime.now().hour)
        weather= data.get("weather","Clear")

        if origin not in LOCATION_COORDS or dest not in LOCATION_COORDS:
            return jsonify({"error":"Unknown location. Use Raipur area locations."}), 400

        # Score every location as a waypoint
        scored = []
        for loc, (lat, lon) in LOCATION_COORDS.items():
            _, dow, month, is_weekend, is_rush, is_night = get_time_features(hour)
            weather_enc = WEATHER_MAP.get(weather, 0)

            t_feat = [[hour, dow, month, is_weekend, is_rush,
                       400, 30, weather_enc, 0]]
            t_pred = int(traffic_model.predict(t_feat)[0])

            c_feat = [[hour, dow, month, is_night, is_weekend, is_rush, 5]]
            c_prob = float(crime_model.predict_proba(c_feat)[0][1])

            a_feat = [[hour, dow, month, is_weekend, is_night, is_rush,
                       1, 60, weather_enc, 0, int(is_night), 0]]
            a_pred = int(accident_model.predict(a_feat)[0])

            safety = max(0, round(100 - t_pred*15 - a_pred*10 - c_prob*30))
            scored.append({
                "location"    : loc,
                "lat"         : lat,
                "lon"         : lon,
                "safety_score": safety,
                "traffic"     : CONGESTION_LABELS[t_pred],
                "crime_risk"  : round(c_prob*100,1)
            })

        scored.sort(key=lambda x: x["safety_score"], reverse=True)

        # Build route: origin → safest midpoints → destination
        origin_data = next(s for s in scored if s["location"]==origin)
        dest_data   = next(s for s in scored if s["location"]==dest)
        midpoints   = [s for s in scored
                       if s["location"] not in [origin, dest]][:2]

        route = [origin_data] + midpoints + [dest_data]
        avg_safety = round(sum(r["safety_score"] for r in route) / len(route))

        return jsonify({
            "origin"         : origin,
            "destination"    : dest,
            "route"          : route,
            "avg_safety_score": avg_safety,
            "recommendation" : "Safe route" if avg_safety>=60 else "Caution advised",
            "total_stops"    : len(route)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ════════════════════════════════════════════════
from flask import render_template

@app.route("/dashboard")
def dashboard():
    return render_template("index.html",
        template_folder="../frontend/templates")

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")