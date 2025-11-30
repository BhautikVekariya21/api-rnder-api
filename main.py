# =============================================================================
#  AQI PREDICTION API - Render Deployment (Gzip Compressed Model)
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from datetime import datetime
from pathlib import Path
import gzip
import tempfile
import os

# =============================================================================
# LOAD MODEL
# =============================================================================
MODEL_DIR = Path("./models")

print("🚀 Starting AQI Prediction API...")
print("=" * 70)

# Load gzipped JSON model
print("📦 Loading compressed model...")
try:
    with gzip.open(MODEL_DIR / "model.json.gz", 'rb') as f:
        model_bytes = f.read()
    
    # Write to temporary file and load
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp.write(model_bytes)
        tmp_path = tmp.name
    
    model = xgb.Booster()
    model.load_model(tmp_path)
    os.unlink(tmp_path)  # Clean up temp file
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Load feature names
try:
    with open(MODEL_DIR / "features.txt", 'r') as f:
        FEATURE_COLS = [line.strip() for line in f.readlines()]
    print(f"✓ Features loaded: {len(FEATURE_COLS)} features")
except Exception as e:
    print(f"❌ Error loading features: {e}")
    raise

# =============================================================================
# CITY DATA (29 Major Indian Cities)
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

# Encodings
CITY_ENC = {c: i for i, c in enumerate(sorted(CITIES.keys()))}
STATES = sorted(set(v['state'] for v in CITIES.values()))
STATE_ENC = {s: i for i, s in enumerate(STATES)}

print(f"✓ Cities loaded: {len(CITIES)} cities across {len(STATES)} states")
print("=" * 70)

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="AQI Prediction API",
    description="Real-time Air Quality Index predictions for major Indian cities",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def aqi_category(aqi):
    """Categorize AQI value with emoji and color"""
    if aqi <= 50:
        return {"cat": "Good", "emoji": "🟢", "color": "#00e400"}
    elif aqi <= 100:
        return {"cat": "Moderate", "emoji": "🟡", "color": "#ffff00"}
    elif aqi <= 150:
        return {"cat": "Unhealthy for Sensitive", "emoji": "🟠", "color": "#ff7e00"}
    elif aqi <= 200:
        return {"cat": "Unhealthy", "emoji": "🔴", "color": "#ff0000"}
    elif aqi <= 300:
        return {"cat": "Very Unhealthy", "emoji": "🟣", "color": "#8f3f97"}
    else:
        return {"cat": "Hazardous", "emoji": "🟤", "color": "#7e0023"}


def fetch_data(lat, lon, days=2):
    """Fetch weather and air quality data from Open-Meteo APIs"""
    try:
        # Weather data
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "relative_humidity_2m,dew_point_2m,wind_gusts_10m,precipitation,pressure_msl,cloud_cover",
                "timezone": "Asia/Kolkata",
                "forecast_days": days
            },
            timeout=30
        ).json()
        
        # Air quality data
        air_quality = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,aerosol_optical_depth",
                "timezone": "Asia/Kolkata",
                "forecast_days": days
            },
            timeout=30
        ).json()
        
        return weather, air_quality
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None


def safe_get(data_dict, key, index, default, total_length):
    """Safely extract value from API response with fallback"""
    try:
        value = data_dict.get(key, [default] * total_length)[index]
        return value if value is not None else default
    except (IndexError, KeyError):
        return default


def prepare_features(weather, air_quality, city):
    """Prepare feature dataframe from API data"""
    city_info = CITIES[city]
    weather_hourly = weather.get('hourly', {})
    air_hourly = air_quality.get('hourly', {})
    n_hours = len(weather_hourly.get('time', []))
    
    if n_hours == 0:
        return None
    
    rows = []
    for i in range(n_hours):
        dt = pd.to_datetime(weather_hourly['time'][i])
        precip = safe_get(weather_hourly, 'precipitation', i, 0, n_hours)
        
        rows.append({
            'datetime': dt,
            'hour': dt.hour,
            'month': dt.month,
            'is_weekend': 1 if dt.dayofweek >= 5 else 0,
            'latitude': city_info['lat'],
            'longitude': city_info['lon'],
            'city_encoded': CITY_ENC.get(city, 0),
            'state_encoded': STATE_ENC.get(city_info['state'], 0),
            'o3_ugm3': safe_get(air_hourly, 'ozone', i, 50, n_hours),
            'pm2_5_ugm3': safe_get(air_hourly, 'pm2_5', i, 50, n_hours),
            'pm10_ugm3': safe_get(air_hourly, 'pm10', i, 80, n_hours),
            'co_ugm3': safe_get(air_hourly, 'carbon_monoxide', i, 500, n_hours),
            'no2_ugm3': safe_get(air_hourly, 'nitrogen_dioxide', i, 30, n_hours),
            'so2_ugm3': safe_get(air_hourly, 'sulphur_dioxide', i, 10, n_hours),
            'dust_ugm3': safe_get(air_hourly, 'dust', i, 10, n_hours),
            'aod': safe_get(air_hourly, 'aerosol_optical_depth', i, 0.3, n_hours),
            'humidity_percent': safe_get(weather_hourly, 'relative_humidity_2m', i, 60, n_hours),
            'dew_point_c': safe_get(weather_hourly, 'dew_point_2m', i, 15, n_hours),
            'wind_gusts_kmh': safe_get(weather_hourly, 'wind_gusts_10m', i, 20, n_hours),
            'precipitation_mm': precip,
            'is_raining': 1 if precip > 0 else 0,
            'heavy_rain': 1 if precip > 7.5 else 0,
            'pressure_msl_hpa': safe_get(weather_hourly, 'pressure_msl', i, 1013, n_hours),
            'cloud_cover_percent': safe_get(weather_hourly, 'cloud_cover', i, 30, n_hours),
        })
    
    return pd.DataFrame(rows)


def predict_aqi(df):
    """Generate AQI predictions from feature dataframe"""
    X = df[FEATURE_COLS].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    return model.predict(dmatrix)


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.get("/")
def root():
    """API root endpoint"""
    return {
        "status": "ok",
        "api": "AQI Prediction API",
        "version": "2.0.0",
        "cities": len(CITIES),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "cities": "/cities",
            "predict": "/predict/{city}"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": "loaded",
        "cities": len(CITIES)
    }


@app.get("/cities")
def get_cities():
    """Get list of available cities"""
    return {
        "total": len(CITIES),
        "cities": [
            {"name": city, "state": info["state"], "lat": info["lat"], "lon": info["lon"]}
            for city, info in sorted(CITIES.items())
        ],
        "states": STATES
    }


@app.get("/predict/{city}")
def predict_city_aqi(city: str, days: int = 2):
    """
    Predict AQI for a specific city
    
    Parameters:
    - city: City name (case-sensitive)
    - days: Number of forecast days (1-5, default 2)
    """
    # Validate city
    if city not in CITIES:
        raise HTTPException(
            status_code=400,
            detail=f"City '{city}' not found. Available cities: {list(CITIES.keys())}"
        )
    
    # Validate days
    days = max(1, min(5, days))
    city_info = CITIES[city]
    
    # Fetch data
    weather, air_quality = fetch_data(city_info['lat'], city_info['lon'], days)
    if not weather or not air_quality:
        raise HTTPException(
            status_code=503,
            detail="Failed to fetch weather/air quality data. Please try again."
        )
    
    # Prepare features
    df = prepare_features(weather, air_quality, city)
    if df is None or len(df) == 0:
        raise HTTPException(
            status_code=500,
            detail="Failed to process data. No valid entries found."
        )
    
    # Generate predictions
    predictions = predict_aqi(df)
    df['aqi'] = predictions
    
    # Build hourly forecast
    hourly_forecast = []
    for _, row in df.iterrows():
        category = aqi_category(row['aqi'])
        hourly_forecast.append({
            "datetime": row['datetime'].isoformat(),
            "hour": int(row['hour']),
            "aqi": round(float(row['aqi']), 1),
            "category": category["cat"],
            "emoji": category["emoji"],
            "color": category["color"],
            "pm2_5": round(float(row['pm2_5_ugm3']), 1),
            "pm10": round(float(row['pm10_ugm3']), 1),
            "o3": round(float(row['o3_ugm3']), 1),
            "no2": round(float(row['no2_ugm3']), 1),
            "so2": round(float(row['so2_ugm3']), 1),
            "co": round(float(row['co_ugm3']), 1),
            "humidity": round(float(row['humidity_percent']), 1),
            "wind_gusts": round(float(row['wind_gusts_kmh']), 1)
        })
    
    # Build daily summary
    df['date'] = df['datetime'].dt.date
    daily_summary = []
    for date, group in df.groupby('date'):
        daily_cat = aqi_category(group['aqi'].mean())
        daily_summary.append({
            "date": str(date),
            "avg_aqi": round(float(group['aqi'].mean()), 1),
            "max_aqi": round(float(group['aqi'].max()), 1),
            "min_aqi": round(float(group['aqi'].min()), 1),
            "category": daily_cat["cat"],
            "emoji": daily_cat["emoji"],
            "color": daily_cat["color"]
        })
    
    # Overall summary
    avg_aqi = float(predictions.mean())
    overall_category = aqi_category(avg_aqi)
    
    return {
        "success": True,
        "city": city,
        "state": city_info['state'],
        "coordinates": {
            "lat": city_info['lat'],
            "lon": city_info['lon']
        },
        "forecast_days": days,
        "generated_at": datetime.now().isoformat(),
        "hourly": hourly_forecast,
        "daily": daily_summary,
        "summary": {
            "avg_aqi": round(avg_aqi, 1),
            "max_aqi": round(float(predictions.max()), 1),
            "min_aqi": round(float(predictions.min()), 1),
            "category": overall_category["cat"],
            "emoji": overall_category["emoji"],
            "color": overall_category["color"],
            "total_hours": len(hourly_forecast)
        }
    }


# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)