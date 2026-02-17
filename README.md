# 🌫️ Karachi Air Quality Prediction System

> Real-time 3-day Air Quality Index (AQI) predictions for Karachi using Machine Learning and MLOps

---

## 📌 Overview

The **Karachi Air Quality Prediction System** predicts AQI for the next 3 days using historical and real-time pollution data.

### ✅ Key Features
- Automated hourly data collection from OpenWeatherMap API
- Rolling & lag feature engineering
- Training & comparison of CatBoost, XGBoost, and Random Forest
- CI/CD automation using GitHub Actions
- Interactive Streamlit dashboard
- Hopsworks Feature Store & Model Registry integration

---

## 🏗️ Architecture

```
GitHub Actions (CI/CD)
│
├── Hourly Pipeline
│   ├── Fetch AQI from OpenWeatherMap
│   ├── Clean & validate data
│   └── Upload to Hopsworks Feature Store
│
└── Daily Training Pipeline
    ├── Fetch full dataset
    ├── Feature Engineering
    ├── Train Models (CatBoost | XGBoost | RF)
    ├── Compare Metrics
    └── Upload Best Model to Registry

Streamlit Dashboard
│
└── Real-time AQI predictions & visualization
```

---

## 📊 Dataset

- **Source:** OpenWeatherMap Air Pollution API  
- **Location:** Karachi, Pakistan (24.8607°N, 67.0011°E)  
- **Collection Frequency:** Hourly  
- **Historical Data:** 6 months (Aug 2025 – Feb 2026, ~4,200 records)

### Pollutants Collected
- AQI
- PM10
- PM2.5
- CO
- O₃

---

## ⚙️ Features

### 🔹 Selected Features (12)

1. `aqi_rolling_max_24h`
2. `pm10`
3. `pm25`
4. `aqi`
5. `aqi_rolling_mean_3h`
6. `aqi_lag_1h`
7. `aqi_rolling_mean_6h`
8. `co`
9. `aqi_rolling_mean_12h`
10. `aqi_lag_3h`
11. `o3`
12. `aqi_lag_6h`

### 🎯 Target Variables

- `target_aqi_1d` (24 hours ahead)
- `target_aqi_2d` (48 hours ahead)
- `target_aqi_3d` (72 hours ahead)

---

## 🤖 Model Performance

| Model | Train R² | Test R² | MAE | Overfitting |
|-------|----------|---------|-----|-------------|
| **CatBoost** | **0.9120** | **0.8582** | **10.02** | **0.0538** |
| XGBoost | 0.9050 | 0.8450 | 10.85 | 0.0600 |
| Random Forest | 0.8980 | 0.8320 | 11.52 | 0.0660 |

### 🏆 Best Model
**CatBoost** – 85.82% Test R² Score

---

## 🚀 Installation

### 🔹 Prerequisites
- Python 3.11+
- Hopsworks account
- OpenWeatherMap API key

### 🔹 Setup

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/MuhammadUzair17/aqi_prediction.git
cd aqi_prediction
```

#### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
```

Activate:

Windows:
```bash
venv\Scripts\activate
```

Linux / Mac:
```bash
source venv/bin/activate
```

#### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4️⃣ Configure Environment Variables

Create `.env` file:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

---

## ▶️ Usage

### Run Streamlit Dashboard
```bash
streamlit run app.py
```

### Run Pipelines Manually

Fetch Current Data:
```bash
python pipelines/feature_pipeline.py
```

Train Models:
```bash
python pipelines/train_pipeline.py
```

Upload Historical Data (One-time):
```bash
python pipelines/re_upload_raw.py
```

---

## 🔄 CI/CD Pipelines

### ⏰ Hourly Feature Pipeline
- Runs every hour
- Fetches AQI from API
- Cleans & validates data
- Removes duplicates
- Appends to Feature Store

File:
```
.github/workflows/hourly_features.yml
```

---

### 🌙 Daily Training Pipeline
- Runs daily at 2:00 AM UTC
- Fetches full dataset
- Engineers rolling & lag features
- Trains 3 models
- Compares metrics
- Uploads best model to registry

File:
```
.github/workflows/daily_training.yml
```

---

## 🔐 GitHub Secrets Setup

Go to:

```
Settings → Secrets and variables → Actions
```

Add:

- `HOPSWORKS_API_KEY`
- `OPENWEATHER_API_KEY`

---

## 📂 Project Structure

```
aqi_prediction/
│
├── .github/
│   └── workflows/
│       ├── hourly_features.yml
│       └── daily_training.yml
│
├── pipelines/
│   ├── feature_pipeline.py
│   ├── train_pipeline.py
│   └── re_upload_raw.py
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## 🛠️ Technologies

### Data & ML
- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost

### MLOps
- Hopsworks (Feature Store & Model Registry)
- GitHub Actions
- PyArrow
- Confluent Kafka

### Visualization
- Streamlit
- Plotly

### APIs
- OpenWeatherMap API
- Hopsworks API

---

## 📈 Results

- 🎯 Test R²: **85.82%**
- 📉 MAE: **10.02 AQI units**
- 🔒 Overfitting Gap: **0.0538**
- 📊 Data Growth: 24 new rows/day
- 🔄 Daily automated retraining

---

## 🖥️ Dashboard Features

- Real-time AQI display
- Color-coded health categories
- Pollutant monitoring (PM2.5, PM10, CO, O₃)
- 3-day predictions
- Model selector
- Interactive charts
- Performance metrics display

---

## 👨‍💻 Author

**Muhammad Uzair**  
10Pearls Data Science Internship  

GitHub: https://github.com/MuhammadUzair17  
Project: https://github.com/MuhammadUzair17/aqi_prediction  

---

## ⭐ Support

If you find this project useful, consider giving it a star ⭐
