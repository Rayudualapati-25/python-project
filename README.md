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

### Prerequisites

1. **Download Dataset** from Kaggle:
   - Visit: https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data
   - Download `train.csv` and `test.csv`
   - Place them in `data/raw/` directory

### Setup Instructions

#### 1. Clone Repository

```bash
git clone https://github.com/Rayudualapati-25/python-project.git
cd python-project
```

#### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Place Dataset Files

```bash
# Create data directory if it doesn't exist
mkdir -p data/raw

# Move your downloaded CSV files
mv ~/Downloads/train.csv data/raw/
mv ~/Downloads/test.csv data/raw/
```

#### 5. Train the Model

```bash
python train.py
```

This will:
- Load and process `data/raw/train.csv`
- Train XGBoost model with feature engineering
- Save trained pipeline to `taxi_fare_pipeline.pkl`
- Display training metrics and validation results

Training typically takes 5-15 minutes depending on your hardware.

#### 6. Generate Predictions (Optional)

```bash
python predict.py
```

This will:
- Load the trained model
- Process `data/raw/test.csv`
- Generate predictions in `data/processed/predictions.csv`

#### 7. Run the Web App

```bash
streamlit run app.py
```

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

Expected performance after training on the full dataset:

- **R² Score:** ~86-88%
- **RMSE:** $3-4
- **Training Time:** 5-15 minutes (depends on hardware and sample size)

Performance metrics will be displayed after running `train.py`.

### Sample Output
```
MODEL EVALUATION
==================================================
Training R² Score: 0.8756
Validation R² Score: 0.8623
Training RMSE: $3.12
Validation RMSE: $3.59
```

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
python-project/
├── app.py                          # Streamlit web application
├── train.py                        # Model training script
├── predict.py                      # Batch prediction script
├── taxi_fare_pipeline.pkl          # Trained ML model (generated)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
└── data/
    ├── raw/                        # Place train.csv and test.csv here
    │   ├── train.csv              # Training dataset (download from Kaggle)
    │   └── test.csv               # Test dataset (download from Kaggle)
    └── processed/                  # Generated predictions
        └── predictions.csv        # Model predictions (generated)
```

## 🔧 Technical Details

### Model Architecture
- **Algorithm:** XGBoost Regressor
- **Features:** 20 engineered features
- **Hyperparameters:**
  - n_estimators: 300
  - max_depth: 7
  - learning_rate: 0.1
  - subsample: 0.8
  - colsample_bytree: 0.8

### Feature Engineering Pipeline
1. **DateTime Features** - Extract year, month, day, weekday, hour from pickup_datetime
2. **Trip Distance** - Calculate haversine distance between pickup and dropoff
3. **Landmark Distances** - Distance to JFK, LGA, EWR, Met Museum, WTC (10 features)
4. **Outlier Removal** - Filter invalid coordinates, unreasonable fares
5. **Feature Selection** - Select 20 most relevant features

### Training Pipeline (`train.py`)
```python
# Load data with optional sampling
df = pd.read_csv('data/raw/train.csv')

# Feature engineering pipeline
Pipeline([
    ('datetime_features', DatetimeFeatureExtractor()),
    ('distance', DistanceCalculator()),
    ('landmarks', LandmarkDistanceCalculator()),
    ('outlier_removal', OutlierRemover()),
    ('feature_selection', FeatureSelector())
])

# Train XGBoost
model = XGBRegressor(n_estimators=300, max_depth=7, ...)
model.fit(X_train, y_train)
```

### Prediction Pipeline (`predict.py`)
```python
# Load trained model
pipeline_data = pickle.load('taxi_fare_pipeline.pkl')

# Transform test data
X_test = pipeline_data['feature_pipeline'].transform(test_df)

# Generate predictions
predictions = pipeline_data['model'].predict(X_test)
```

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
