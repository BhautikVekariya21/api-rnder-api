# =============================================================================
#  AQI PREDICTION API - Render (Using PKL directly)
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import joblib
from datetime import datetime
from pathlib import Path
import os

# =============================================================================
# DEFINE WRAPPER CLASS (MUST BE BEFORE LOADING PKL)
# =============================================================================
class OptimizedXGBoostWrapper:
    """Wrapper class - must match the one used when saving"""
    def __init__(self, model, feature_names, best_iteration):
        self.model = model
        self.feature_names = feature_names
        self.best_iteration = best_iteration
        self._estimator_type = "regressor"
        
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        X = np.asarray(X, dtype=np.float32)
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dmatrix, iteration_range=(0, self.best_iteration + 1))

# =============================================================================
# LOAD MODEL
# =============================================================================
MODEL_DIR = Path("./models")

print("🚀 Starting AQI API...")

# Load PKL directly (6.6 MB - smallest!)
model = joblib.load(MODEL_DIR / "xgboost_improved_lzma.pkl")
print("✓ Model loaded: xgboost_improved_lzma.pkl")

# Get feature names from the wrapper
FEATURE_COLS = model.feature_names
print(f"✓ Features: {len(FEATURE_COLS)}")

# =============================================================================
# CITY DATA
# =============================================================================
CITIES = {
    "Agartala": {"lat": 23.8315, "lon": 91.2868, "state": "Tripura"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "Aizawl": {"lat": 23.7271, "lon": 92.7176, "state": "Mizoram"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245, "state": "Odisha"},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "state": "Punjab"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Dehradun": {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand"},
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "Gangtok": {"lat": 27.3389, "lon": 88.6065, "state": "Sikkim"},
    "Gurugram": {"lat": 28.4595, "lon": 77.0266, "state": "Haryana"},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362, "state": "Assam"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Imphal": {"lat": 24.8170, "lon": 93.9368, "state": "Manipur"},
    "Itanagar": {"lat": 27.0844, "lon": 93.6053, "state": "Arunachal Pradesh"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Kohima": {"lat": 25.6751, "lon": 94.1086, "state": "Nagaland"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Panaji": {"lat": 15.4909, "lon": 73.8278, "state": "Goa"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "Raipur": {"lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh"},
    "Ranchi": {"lat": 23.3441, "lon": 85.3096, "state": "Jharkhand"},
    "Shillong": {"lat": 25.5788, "lon": 91.8933, "state": "Meghalaya"},
    "Shimla": {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh"},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366, "state": "Kerala"},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
}

CITY_ENC = {c: i for i, c in enumerate(sorted(CITIES.keys()))}
STATES = sorted(set(v['state'] for v in CITIES.values()))
STATE_ENC = {s: i for i, s in enumerate(STATES)}

print(f"✓ Cities: {len(CITIES)}")

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title="AQI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# HELPERS
# =============================================================================
def aqi_category(aqi):
    if aqi <= 50: return {"cat": "Good", "emoji": "🟢", "color": "#00e400"}
    if aqi <= 100: return {"cat": "Moderate", "emoji": "🟡", "color": "#ffff00"}
    if aqi <= 150: return {"cat": "Unhealthy for Sensitive", "emoji": "🟠", "color": "#ff7e00"}
    if aqi <= 200: return {"cat": "Unhealthy", "emoji": "🔴", "color": "#ff0000"}
    if aqi <= 300: return {"cat": "Very Unhealthy", "emoji": "🟣", "color": "#8f3f97"}
    return {"cat": "Hazardous", "emoji": "🟤", "color": "#7e0023"}


def fetch_data(lat, lon, days=2):
    try:
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "relative_humidity_2m,dew_point_2m,wind_gusts_10m,precipitation,pressure_msl,cloud_cover",
                "timezone": "Asia/Kolkata", "forecast_days": days
            },
            timeout=30
        ).json()
        
        a = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,aerosol_optical_depth",
                "timezone": "Asia/Kolkata", "forecast_days": days
            },
            timeout=30
        ).json()
        
        return w, a
    except:
        return None, None


def safe_get(d, k, i, default, n):
    try:
        v = d.get(k, [default]*n)[i]
        return v if v is not None else default
    except:
        return default


def prepare(w, a, city):
    info = CITIES[city]
    wh = w.get('hourly', {})
    ah = a.get('hourly', {})
    n = len(wh.get('time', []))
    
    if n == 0:
        return None
    
    rows = []
    for i in range(n):
        dt = pd.to_datetime(wh['time'][i])
        p = safe_get(wh, 'precipitation', i, 0, n)
        
        rows.append({
            'datetime': dt,
            'hour': dt.hour,
            'o3_ugm3': safe_get(ah, 'ozone', i, 50, n),
            'pressure_msl_hpa': safe_get(wh, 'pressure_msl', i, 1013, n),
            'heavy_rain': 1 if p > 7.5 else 0,
            'co_ugm3': safe_get(ah, 'carbon_monoxide', i, 500, n),
            'latitude': info['lat'],
            'humidity_percent': safe_get(wh, 'relative_humidity_2m', i, 60, n),
            'city_encoded': CITY_ENC.get(city, 0),
            'so2_ugm3': safe_get(ah, 'sulphur_dioxide', i, 10, n),
            'precipitation_mm': p,
            'dust_ugm3': safe_get(ah, 'dust', i, 10, n),
            'pm2_5_ugm3': safe_get(ah, 'pm2_5', i, 50, n),
            'wind_gusts_kmh': safe_get(wh, 'wind_gusts_10m', i, 20, n),
            'aod': safe_get(ah, 'aerosol_optical_depth', i, 0.3, n),
            'state_encoded': STATE_ENC.get(info['state'], 0),
            'pm10_ugm3': safe_get(ah, 'pm10', i, 80, n),
            'longitude': info['lon'],
            'is_raining': 1 if p > 0 else 0,
            'cloud_cover_percent': safe_get(wh, 'cloud_cover', i, 30, n),
            'is_weekend': 1 if dt.dayofweek >= 5 else 0,
            'dew_point_c': safe_get(wh, 'dew_point_2m', i, 15, n),
            'no2_ugm3': safe_get(ah, 'nitrogen_dioxide', i, 30, n),
            'month': dt.month
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/")
def root():
    return {
        "status": "ok",
        "api": "AQI Prediction",
        "version": "2.0.0",
        "model_size": "6.6 MB",
        "cities": len(CITIES),
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "time": datetime.now().isoformat()}


@app.get("/cities")
def cities():
    return {
        "total": len(CITIES),
        "cities": [{"name": c, "state": v["state"]} for c, v in sorted(CITIES.items())]
    }


@app.get("/predict/{city}")
def predict(city: str, days: int = 2):
    if city not in CITIES:
        raise HTTPException(400, f"City not found. Available: {list(CITIES.keys())}")
    
    days = max(1, min(5, days))
    info = CITIES[city]
    
    w, a = fetch_data(info['lat'], info['lon'], days)
    if not w or not a:
        raise HTTPException(503, "Failed to fetch data")
    
    df = prepare(w, a, city)
    if df is None or len(df) == 0:
        raise HTTPException(500, "No data")
    
    # Use wrapper's predict method directly
    X = df[FEATURE_COLS].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0)
    preds = model.predict(X)
    df['aqi'] = preds
    
    hourly = []
    for _, r in df.iterrows():
        c = aqi_category(r['aqi'])
        hourly.append({
            "datetime": r['datetime'].isoformat(),
            "hour": int(r['hour']),
            "aqi": round(float(r['aqi']), 1),
            "category": c["cat"],
            "emoji": c["emoji"],
            "color": c["color"],
            "pm2_5": round(float(r['pm2_5_ugm3']), 1),
            "pm10": round(float(r['pm10_ugm3']), 1),
            "o3": round(float(r['o3_ugm3']), 1),
            "humidity": round(float(r['humidity_percent']), 1)
        })
    
    avg = float(preds.mean())
    c = aqi_category(avg)
    
    df['date'] = df['datetime'].dt.date
    daily = []
    for date, g in df.groupby('date'):
        dc = aqi_category(g['aqi'].mean())
        daily.append({
            "date": str(date),
            "avg": round(float(g['aqi'].mean()), 1),
            "max": round(float(g['aqi'].max()), 1),
            "min": round(float(g['aqi'].min()), 1),
            "category": dc["cat"],
            "emoji": dc["emoji"]
        })
    
    return {
        "success": True,
        "city": city,
        "state": info['state'],
        "lat": info['lat'],
        "lon": info['lon'],
        "days": days,
        "generated": datetime.now().isoformat(),
        "hourly": hourly,
        "daily": daily,
        "summary": {
            "avg": round(avg, 1),
            "max": round(float(preds.max()), 1),
            "min": round(float(preds.min()), 1),
            "category": c["cat"],
            "emoji": c["emoji"],
            "color": c["color"],
            "hours": len(hourly)
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)