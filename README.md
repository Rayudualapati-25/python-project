# 🚕 NYC Taxi Fare Predictor

A beautiful, modern web application for predicting New York City taxi fares using machine learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green.svg)

## ✨ Features

- 🎯 **AI-Powered Predictions** - 86.23% accuracy using XGBoost
- 🗺️ **Interactive Maps** - Visual route display with Folium
- 📍 **NYC Landmarks** - Quick selection of popular locations
- 💰 **Detailed Breakdown** - See fare components and estimates
- 🎨 **Modern UI** - Beautiful gradient design with smooth animations
- 📱 **Responsive** - Works on desktop and mobile

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

### 3. Open in Browser

The app will automatically open at `http://localhost:8501`

## 📋 Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- folium
- streamlit-folium

## 🎮 How to Use

1. **Select Pickup Location** - Choose from NYC landmarks or enter custom coordinates
2. **Select Dropoff Location** - Choose your destination
3. **Set Date & Time** - Pick your desired pickup time
4. **Choose Passengers** - Select number of passengers (1-6)
5. **Click Predict** - Get your estimated fare!

## 🏆 Model Performance

- **R² Score:** 86.23%
- **RMSE:** $3.59
- **MAE:** $1.56
- **MAPE:** 16.38%

The model was trained on 5.5 million NYC taxi trips and uses 21 engineered features including:
- Trip distance
- Pickup/dropoff coordinates
- Date and time features
- Distance from major landmarks (JFK, LGA, EWR, Times Square, etc.)

## 🎨 Features Showcase

### 🌈 Modern UI Design
- Gradient backgrounds (Purple to Blue theme)
- Glassmorphism cards
- Smooth animations and transitions
- Responsive layout

### 📊 Interactive Visualizations
- Real-time route mapping
- Distance calculations (km and miles)
- Fare breakdown charts
- Estimated trip duration

### 🏙️ NYC Landmarks Included
- Times Square
- Statue of Liberty
- Central Park
- JFK Airport
- LaGuardia Airport
- Newark Airport
- Metropolitan Museum
- World Trade Center
- Brooklyn Bridge
- Empire State Building

## 📁 Project Structure

```
NYC-TAXI-FARE-PREDICTION/
├── app.py                          # Streamlit application
├── taxi_fare_pipeline.pkl          # Trained ML model
├── nyc-taxi-fare-pipeline.ipynb    # Model training notebook
├── data-visualization.ipynb        # Data exploration notebook
├── requirements.txt                # Python dependencies
├── train.csv                       # Training data
├── test.csv                        # Test data
└── README.md                       # This file
```

## 🔧 Technical Details

### Model Architecture
- **Algorithm:** XGBoost Regressor
- **Features:** 21 engineered features
- **Training Data:** 4.3M samples
- **Validation Data:** 1.1M samples

### Feature Engineering Pipeline
1. DateTime feature extraction (year, month, day, weekday, hour)
2. Haversine distance calculation
3. Landmark distance calculations (5 major NYC landmarks)
4. Outlier removal
5. Feature selection

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Built with ❤️ using Streamlit and XGBoost

## 🙏 Acknowledgments

- Dataset: NYC Taxi & Limousine Commission
- Framework: Streamlit
- ML Library: XGBoost
- Mapping: Folium

---

**Made with 🚕 in NYC**
