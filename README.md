# AQI Prediction API - Deployment Guide

Real-time Air Quality Index (AQI) prediction API for 29 major Indian cities using XGBoost ML model.

## 🚀 Quick Start

### 1. Convert Your Model to Gzip Format

Run this script **locally** where you have your `xgboost_improved_lzma.pkl` file:

```python
import joblib
import xgboost as xgb
from pathlib import Path
import json
import gzip

# Define paths
model_dir = Path(r"D:\Downloads\aqi\featured\models\final_optimized")
output_dir = model_dir / "render_deploy"
output_dir.mkdir(exist_ok=True)

# Load PKL file
class OptimizedXGBoostWrapper:
    def __init__(self, model, feature_names, best_iteration):
        self.model = model
        self.feature_names = feature_names
        self.best_iteration = best_iteration

pkl_path = model_dir / 'xgboost_improved_lzma.pkl'
wrapped = joblib.load(pkl_path)
booster = wrapped.model
feature_names = wrapped.feature_names

# Save feature names
with open(output_dir / 'features.txt', 'w') as f:
    for feat in feature_names:
        f.write(f"{feat}\n")

# Save model as JSON
json_path = output_dir / 'model.json'
booster.save_model(str(json_path))

# Compress with gzip
gz_path = output_dir / 'model.json.gz'
with open(json_path, 'rb') as f_in:
    with gzip.open(gz_path, 'wb', compresslevel=9) as f_out:
        f_out.write(f_in.read())

# Clean up large JSON
json_path.unlink()

print(f"✅ Created: model.json.gz ({gz_path.stat().st_size / (1024*1024):.1f} MB)")
print(f"✅ Created: features.txt")
print(f"\n📁 Files saved to: {output_dir}")
```

### 2. Create Repository Structure

```
your-repo/
├── main.py              # API code (already created)
├── requirements.txt     # Dependencies
├── render.yaml          # Render config
├── Dockerfile           # Docker config
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── models/
    ├── model.json.gz   # ~8 MB (gzipped model) ✅
    └── features.txt    # Feature names
```

### 3. Push to GitHub

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: AQI Prediction API"

# Create GitHub repository and push
git remote add origin https://github.com/YOUR_USERNAME/aqi-api.git
git branch -M main
git push -u origin main
```

### 4. Deploy to Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` and configure everything
5. Click "Create Web Service"

**Or manually configure:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/health`

## 📊 API Endpoints

### GET `/`
Root endpoint with API info

```json
{
  "status": "ok",
  "api": "AQI Prediction API",
  "version": "2.0.0",
  "cities": 29
}
```

### GET `/health`
Health check endpoint

```json
{
  "status": "healthy",
  "timestamp": "2025-11-30T...",
  "model": "loaded",
  "cities": 29
}
```

### GET `/cities`
Get list of all available cities

```json
{
  "total": 29,
  "cities": [
    {"name": "Agartala", "state": "Tripura", "lat": 23.8315, "lon": 91.2868},
    ...
  ],
  "states": ["Andhra Pradesh", ...]
}
```

### GET `/predict/{city}?days=2`
Get AQI predictions for a city

**Parameters:**
- `city` (required): City name (case-sensitive)
- `days` (optional): Forecast days (1-5, default: 2)

**Example Request:**
```bash
curl https://your-api.onrender.com/predict/Delhi?days=3
```

**Example Response:**
```json
{
  "success": true,
  "city": "Delhi",
  "state": "Delhi",
  "coordinates": {"lat": 28.6139, "lon": 77.2090},
  "forecast_days": 3,
  "generated_at": "2025-11-30T12:00:00",
  "hourly": [
    {
      "datetime": "2025-11-30T00:00:00",
      "hour": 0,
      "aqi": 156.3,
      "category": "Unhealthy for Sensitive",
      "emoji": "🟠",
      "color": "#ff7e00",
      "pm2_5": 65.2,
      "pm10": 120.5,
      "o3": 45.8,
      "humidity": 68.5
    },
    ...
  ],
  "daily": [
    {
      "date": "2025-11-30",
      "avg_aqi": 152.4,
      "max_aqi": 178.2,
      "min_aqi": 125.6,
      "category": "Unhealthy for Sensitive",
      "emoji": "🟠"
    },
    ...
  ],
  "summary": {
    "avg_aqi": 150.2,
    "max_aqi": 185.6,
    "min_aqi": 120.3,
    "category": "Unhealthy for Sensitive",
    "emoji": "🟠",
    "color": "#ff7e00",
    "total_hours": 72
  }
}
```

## 🏙️ Supported Cities (29)

| City | State | City | State |
|------|-------|------|-------|
| Agartala | Tripura | Kohima | Nagaland |
| Ahmedabad | Gujarat | Kolkata | West Bengal |
| Aizawl | Mizoram | Lucknow | Uttar Pradesh |
| Bengaluru | Karnataka | Mumbai | Maharashtra |
| Bhopal | Madhya Pradesh | Panaji | Goa |
| Bhubaneswar | Odisha | Patna | Bihar |
| Chandigarh | Punjab | Raipur | Chhattisgarh |
| Chennai | Tamil Nadu | Ranchi | Jharkhand |
| Dehradun | Uttarakhand | Shillong | Meghalaya |
| Delhi | Delhi | Shimla | Himachal Pradesh |
| Gangtok | Sikkim | Thiruvananthapuram | Kerala |
| Gurugram | Haryana | Visakhapatnam | Andhra Pradesh |
| Guwahati | Assam | | |
| Hyderabad | Telangana | | |
| Imphal | Manipur | | |
| Itanagar | Arunachal Pradesh | | |
| Jaipur | Rajasthan | | |

## 📊 AQI Categories

| AQI Range | Category | Color | Health Impact |
|-----------|----------|-------|---------------|
| 0-50 | 🟢 Good | Green | Minimal impact |
| 51-100 | 🟡 Moderate | Yellow | Acceptable |
| 101-150 | 🟠 Unhealthy for Sensitive | Orange | Sensitive groups affected |
| 151-200 | 🔴 Unhealthy | Red | Everyone affected |
| 201-300 | 🟣 Very Unhealthy | Purple | Health alert |
| 301+ | 🟤 Hazardous | Maroon | Emergency conditions |

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8000

# Access API
open http://localhost:8000/docs
```

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/health

# List cities
curl http://localhost:8000/cities

# Get prediction
curl http://localhost:8000/predict/Mumbai?days=2
```

## 📦 Model Information

- **Type:** XGBoost Regressor
- **Input Features:** 24 environmental & temporal features
- **Format:** Gzip-compressed JSON (~8 MB)
- **Prediction:** Hourly AQI values for 1-5 days

### Features Used:
- Weather: Temperature, humidity, precipitation, wind, pressure, cloud cover
- Air Quality: PM2.5, PM10, O₃, NO₂, SO₂, CO, dust, AOD
- Location: Latitude, longitude, city, state
- Temporal: Hour, month, day of week

## 🚨 Troubleshooting

### Model Loading Fails
```python
# Verify files exist
ls -lh models/
# Should show: model.json.gz (~8 MB) and features.txt
```

### API Returns 503
- Check if Open-Meteo APIs are accessible
- Verify internet connection from server
- Check API rate limits

### Prediction Errors
- Ensure city name is exactly as listed (case-sensitive)
- Check `days` parameter is between 1-5
- Verify all model features are present

## 📝 License

MIT License - Feel free to use this API for your projects!

## 🤝 Contributing

Pull requests welcome! Please ensure:
- Code follows existing style
- All tests pass
- Documentation is updated

## 📧 Support

For issues or questions:
- Open a GitHub issue
- Check Render logs for deployment issues
- Verify model files are uploaded correctly

---

**Built with:** FastAPI • XGBoost • Open-Meteo APIs • Render