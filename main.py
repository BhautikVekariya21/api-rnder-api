# =============================================================================
#  AQI PREDICTION API - Render Deployment
#  Fast, Production-Ready API
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_DIR = Path("./models")
PORT = int(os.getenv("PORT", 10000))

# =============================================================================
# STARTUP - LOAD MODEL
# =============================================================================
print("=" * 60)
print("🚀 AQI PREDICTION API - Starting on Render...")
print("=" * 60)

# Load XGBoost model
model_path = MODEL_DIR / "xgboost_optimized.json"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = xgb.XGBRegressor()
model.load_model(str(model_path))
print(f"✓ Model loaded: {model_path.name}")

# Load feature names
feature_path = MODEL_DIR / "feature_names.txt"
with open(feature_path, 'r') as f:
    FEATURE_COLS = [line.strip() for line in f.readlines()]
print(f"✓ Features loaded: {len(FEATURE_COLS)}")

# =============================================================================
# CITY DATA (29 Indian Cities)
# =============================================================================
CITIES_DATA = {
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

# Create encodings (must match training)
CITY_ENCODING = {city: idx for idx, city in enumerate(sorted(CITIES_DATA.keys()))}
UNIQUE_STATES = sorted(set(info['state'] for info in CITIES_DATA.values()))
STATE_ENCODING = {state: idx for idx, state in enumerate(UNIQUE_STATES)}

print(f"✓ Cities loaded: {len(CITIES_DATA)}")
print("=" * 60)

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================
app = FastAPI(
    title="AQI Prediction API",
    description="XGBoost-based Air Quality Index Prediction for Indian Cities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Allow all origins
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
def get_aqi_category(aqi: float) -> Dict[str, str]:
    """Get AQI category and color"""
    if aqi <= 50:
        return {"category": "Good", "emoji": "🟢", "color": "#00e400"}
    elif aqi <= 100:
        return {"category": "Moderate", "emoji": "🟡", "color": "#ffff00"}
    elif aqi <= 150:
        return {"category": "Unhealthy for Sensitive Groups", "emoji": "🟠", "color": "#ff7e00"}
    elif aqi <= 200:
        return {"category": "Unhealthy", "emoji": "🔴", "color": "#ff0000"}
    elif aqi <= 300:
        return {"category": "Very Unhealthy", "emoji": "🟣", "color": "#8f3f97"}
    else:
        return {"category": "Hazardous", "emoji": "🟤", "color": "#7e0023"}


def fetch_weather(lat: float, lon: float, days: int = 2) -> Optional[dict]:
    """Fetch weather data from Open-Meteo API"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "relative_humidity_2m,dew_point_2m,wind_gusts_10m,precipitation,pressure_msl,cloud_cover",
        "timezone": "Asia/Kolkata",
        "forecast_days": days
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        print(f"Weather API error: {response.status_code}")
        return None
    except Exception as e:
        print(f"Weather API exception: {e}")
        return None


def fetch_air_quality(lat: float, lon: float, days: int = 2) -> Optional[dict]:
    """Fetch air quality data from Open-Meteo API"""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,aerosol_optical_depth",
        "timezone": "Asia/Kolkata",
        "forecast_days": days
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        print(f"AQ API error: {response.status_code}")
        return None
    except Exception as e:
        print(f"AQ API exception: {e}")
        return None


def safe_get(data: dict, key: str, index: int, default: float, n_hours: int) -> float:
    """Safely extract value from API response"""
    try:
        values = data.get(key, [default] * n_hours)
        if index < len(values):
            value = values[index]
            return value if value is not None else default
        return default
    except:
        return default


def prepare_features(weather: dict, aq: dict, city_name: str) -> Optional[pd.DataFrame]:
    """Prepare features for model prediction"""
    city_info = CITIES_DATA[city_name]
    w = weather.get('hourly', {})
    a = aq.get('hourly', {})
    n = len(w.get('time', []))
    
    if n == 0:
        return None
    
    records = []
    for i in range(n):
        dt = pd.to_datetime(w['time'][i])
        precip = safe_get(w, 'precipitation', i, 0, n)
        
        record = {
            'datetime': dt,
            'hour': dt.hour,
            'o3_ugm3': safe_get(a, 'ozone', i, 50, n),
            'pressure_msl_hpa': safe_get(w, 'pressure_msl', i, 1013, n),
            'heavy_rain': 1 if precip > 7.5 else 0,
            'co_ugm3': safe_get(a, 'carbon_monoxide', i, 500, n),
            'latitude': city_info['lat'],
            'humidity_percent': safe_get(w, 'relative_humidity_2m', i, 60, n),
            'city_encoded': CITY_ENCODING.get(city_name, 0),
            'so2_ugm3': safe_get(a, 'sulphur_dioxide', i, 10, n),
            'precipitation_mm': precip,
            'dust_ugm3': safe_get(a, 'dust', i, 10, n),
            'pm2_5_ugm3': safe_get(a, 'pm2_5', i, 50, n),
            'wind_gusts_kmh': safe_get(w, 'wind_gusts_10m', i, 20, n),
            'aod': safe_get(a, 'aerosol_optical_depth', i, 0.3, n),
            'state_encoded': STATE_ENCODING.get(city_info['state'], 0),
            'pm10_ugm3': safe_get(a, 'pm10', i, 80, n),
            'longitude': city_info['lon'],
            'is_raining': 1 if precip > 0 else 0,
            'cloud_cover_percent': safe_get(w, 'cloud_cover', i, 30, n),
            'is_weekend': 1 if dt.dayofweek >= 5 else 0,
            'dew_point_c': safe_get(w, 'dew_point_2m', i, 15, n),
            'no2_ugm3': safe_get(a, 'nitrogen_dioxide', i, 30, n),
            'month': dt.month
        }
        records.append(record)
    
    return pd.DataFrame(records)


def make_predictions(df: pd.DataFrame) -> np.ndarray:
    """Make predictions using XGBoost model"""
    X = df[FEATURE_COLS].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0)
    return model.predict(X)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "status": "healthy",
        "api": "AQI Prediction API",
        "version": "1.0.0",
        "model": "XGBoost",
        "cities_available": len(CITIES_DATA),
        "features": len(FEATURE_COLS),
        "endpoints": {
            "health": "GET /health",
            "cities": "GET /cities",
            "predict": "GET /predict/{city_name}?days=2",
            "docs": "GET /docs"
        },
        "deployed_on": "Render"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/cities")
async def get_cities():
    """Get list of all available cities"""
    cities_list = [
        {
            "name": city,
            "state": info["state"],
            "lat": info["lat"],
            "lon": info["lon"]
        }
        for city, info in sorted(CITIES_DATA.items())
    ]
    
    return {
        "success": True,
        "total": len(cities_list),
        "cities": cities_list
    }


@app.get("/predict/{city_name}")
async def predict_aqi(city_name: str, days: int = 2):
    """
    Predict AQI for a city
    
    - **city_name**: Name of the city (e.g., Delhi, Mumbai)
    - **days**: Forecast days (1-7, default: 2)
    """
    
    # Validate city
    if city_name not in CITIES_DATA:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"City '{city_name}' not found",
                "available_cities": sorted(list(CITIES_DATA.keys()))
            }
        )
    
    # Validate days
    days = max(1, min(7, days))
    
    city_info = CITIES_DATA[city_name]
    
    # Fetch data from Open-Meteo
    weather = fetch_weather(city_info['lat'], city_info['lon'], days)
    aq = fetch_air_quality(city_info['lat'], city_info['lon'], days)
    
    if weather is None:
        raise HTTPException(status_code=503, detail="Failed to fetch weather data")
    
    if aq is None:
        raise HTTPException(status_code=503, detail="Failed to fetch air quality data")
    
    # Prepare features
    df = prepare_features(weather, aq, city_name)
    
    if df is None or len(df) == 0:
        raise HTTPException(status_code=500, detail="Failed to prepare features")
    
    # Make predictions
    predictions = make_predictions(df)
    df['predicted_aqi'] = predictions
    
    # Build hourly response
    hourly = []
    for idx, row in df.iterrows():
        aqi_info = get_aqi_category(row['predicted_aqi'])
        hourly.append({
            "datetime": row['datetime'].isoformat(),
            "hour": int(row['hour']),
            "predicted_aqi": round(float(row['predicted_aqi']), 1),
            "category": aqi_info["category"],
            "emoji": aqi_info["emoji"],
            "color": aqi_info["color"],
            "pollutants": {
                "pm2_5": round(float(row['pm2_5_ugm3']), 1),
                "pm10": round(float(row['pm10_ugm3']), 1),
                "o3": round(float(row['o3_ugm3']), 1),
                "no2": round(float(row['no2_ugm3']), 1),
                "so2": round(float(row['so2_ugm3']), 1),
                "co": round(float(row['co_ugm3']), 1)
            },
            "weather": {
                "humidity": round(float(row['humidity_percent']), 1),
                "wind_gusts": round(float(row['wind_gusts_kmh']), 1),
                "precipitation": round(float(row['precipitation_mm']), 2),
                "cloud_cover": round(float(row['cloud_cover_percent']), 1)
            }
        })
    
    # Calculate summary
    avg_aqi = float(predictions.mean())
    max_aqi = float(predictions.max())
    min_aqi = float(predictions.min())
    aqi_info = get_aqi_category(avg_aqi)
    
    # Daily summary
    df['date'] = df['datetime'].dt.date
    daily_summary = []
    for date, group in df.groupby('date'):
        day_avg = group['predicted_aqi'].mean()
        day_info = get_aqi_category(day_avg)
        daily_summary.append({
            "date": str(date),
            "avg_aqi": round(float(day_avg), 1),
            "max_aqi": round(float(group['predicted_aqi'].max()), 1),
            "min_aqi": round(float(group['predicted_aqi'].min()), 1),
            "category": day_info["category"],
            "emoji": day_info["emoji"]
        })
    
    return {
        "success": True,
        "city": city_name,
        "state": city_info['state'],
        "coordinates": {
            "lat": city_info['lat'],
            "lon": city_info['lon']
        },
        "forecast_days": days,
        "generated_at": datetime.now().isoformat(),
        "predictions": hourly,
        "daily_summary": daily_summary,
        "summary": {
            "average_aqi": round(avg_aqi, 1),
            "max_aqi": round(max_aqi, 1),
            "min_aqi": round(min_aqi, 1),
            "category": aqi_info["category"],
            "emoji": aqi_info["emoji"],
            "color": aqi_info["color"],
            "total_hours": len(hourly)
        }
    }


@app.get("/predict/{city_name}/current")
async def predict_current(city_name: str):
    """Get current AQI prediction (next few hours)"""
    
    if city_name not in CITIES_DATA:
        raise HTTPException(status_code=400, detail=f"City '{city_name}' not found")
    
    # Get 1 day prediction
    result = await predict_aqi(city_name, days=1)
    
    if not result["success"]:
        return result
    
    # Return only first 6 hours
    current_predictions = result["predictions"][:6]
    
    return {
        "success": True,
        "city": city_name,
        "state": result["state"],
        "current_aqi": current_predictions[0] if current_predictions else None,
        "next_hours": current_predictions,
        "summary": result["summary"]
    }


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "message": "An unexpected error occurred"
        }
    )


# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    print(f"\n🌐 Starting server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)