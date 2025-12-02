"""
NYC Taxi Fare Prediction - Model Training Pipeline
This script trains an XGBoost model on the NYC taxi dataset.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Custom Transformer Classes
class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract datetime features: year, month, day, weekday, hour"""
    
    def __init__(self, datetime_col='pickup_datetime'):
        self.datetime_col = datetime_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(X[self.datetime_col]):
            X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors='coerce')
        
        X[f'{self.datetime_col}_year'] = X[self.datetime_col].dt.year
        X[f'{self.datetime_col}_month'] = X[self.datetime_col].dt.month
        X[f'{self.datetime_col}_day'] = X[self.datetime_col].dt.day
        X[f'{self.datetime_col}_weekday'] = X[self.datetime_col].dt.weekday
        X[f'{self.datetime_col}_hour'] = X[self.datetime_col].dt.hour
        return X


class DistanceCalculator(BaseEstimator, TransformerMixin):
    """Calculate haversine distance between pickup and dropoff locations"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['trip_distance'] = self._haversine_distance(
            X['pickup_longitude'], X['pickup_latitude'],
            X['dropoff_longitude'], X['dropoff_latitude']
        )
        return X
    
    @staticmethod
    def _haversine_distance(lon1, lat1, lon2, lat2):
        """Calculate great circle distance in kilometers"""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        
        return km


class LandmarkDistanceCalculator(BaseEstimator, TransformerMixin):
    """Calculate distances from major NYC landmarks"""
    
    def __init__(self):
        # NYC landmark coordinates (longitude, latitude)
        self.landmarks = {
            'jfk': (-73.7781, 40.6413),
            'lga': (-73.8740, 40.7769),
            'ewr': (-74.1745, 40.6895),
            'met': (-73.9632, 40.7794),
            'wtc': (-74.0099, 40.7126)
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for name, (lon, lat) in self.landmarks.items():
            # Distance from pickup location
            X[f'{name}_pickup_distance'] = self._haversine_distance(
                lon, lat, X['pickup_longitude'], X['pickup_latitude']
            )
            # Distance from dropoff location
            X[f'{name}_drop_distance'] = self._haversine_distance(
                lon, lat, X['dropoff_longitude'], X['dropoff_latitude']
            )
        
        return X
    
    @staticmethod
    def _haversine_distance(lon1, lat1, lon2, lat2):
        """Calculate great circle distance in kilometers"""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        
        return km


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Remove outliers and invalid data based on reasonable ranges"""
    
    def __init__(self, 
                 fare_range=(1, 500),
                 lon_range=(-75, -72),
                 lat_range=(40, 42),
                 passenger_range=(1, 6)):
        self.fare_range = fare_range
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.passenger_range = passenger_range
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # Only apply fare filter if fare_amount column exists (training data)
        if 'fare_amount' in X.columns:
            mask = (
                (X['fare_amount'] >= self.fare_range[0]) &
                (X['fare_amount'] <= self.fare_range[1]) &
                (X['pickup_longitude'] >= self.lon_range[0]) & 
                (X['pickup_longitude'] <= self.lon_range[1]) & 
                (X['dropoff_longitude'] >= self.lon_range[0]) & 
                (X['dropoff_longitude'] <= self.lon_range[1]) & 
                (X['pickup_latitude'] >= self.lat_range[0]) & 
                (X['pickup_latitude'] <= self.lat_range[1]) & 
                (X['dropoff_latitude'] >= self.lat_range[0]) & 
                (X['dropoff_latitude'] <= self.lat_range[1]) & 
                (X['passenger_count'] >= self.passenger_range[0]) & 
                (X['passenger_count'] <= self.passenger_range[1])
            )
        else:
            # For test data without fare_amount
            mask = (
                (X['pickup_longitude'] >= self.lon_range[0]) & 
                (X['pickup_longitude'] <= self.lon_range[1]) & 
                (X['dropoff_longitude'] >= self.lon_range[0]) & 
                (X['dropoff_longitude'] <= self.lon_range[1]) & 
                (X['pickup_latitude'] >= self.lat_range[0]) & 
                (X['pickup_latitude'] <= self.lat_range[1]) & 
                (X['dropoff_latitude'] >= self.lat_range[0]) & 
                (X['dropoff_latitude'] <= self.lat_range[1]) & 
                (X['passenger_count'] >= self.passenger_range[0]) & 
                (X['passenger_count'] <= self.passenger_range[1])
            )
        
        return X[mask]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select only the features needed for modeling"""
    
    def __init__(self, feature_columns=None):
        self.feature_columns = feature_columns
    
    def fit(self, X, y=None):
        if self.feature_columns is None:
            # Define default feature columns
            self.feature_columns = [
                'pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
                'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day',
                'pickup_datetime_weekday', 'pickup_datetime_hour', 'trip_distance',
                'jfk_drop_distance', 'lga_drop_distance', 'ewr_drop_distance',
                'met_drop_distance', 'wtc_drop_distance', 'jfk_pickup_distance',
                'lga_pickup_distance', 'ewr_pickup_distance', 'met_pickup_distance',
                'wtc_pickup_distance'
            ]
        return self
    
    def transform(self, X):
        return X[self.feature_columns]


def load_data(filepath, sample_size=None):
    """Load and optionally sample the dataset"""
    print(f"Loading data from {filepath}...")
    
    # Try to read the CSV
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        
        if sample_size and len(df) > sample_size:
            print(f"Sampling {sample_size} rows...")
            df = df.sample(n=sample_size, random_state=42)
        
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {filepath}")
        print("\nPlease ensure train.csv is in the data/raw/ directory.")
        print("You can download it from Kaggle:")
        print("https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data")
        return None
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None


def train_model(train_path='data/raw/train.csv', sample_size=500000):
    """Train the taxi fare prediction model"""
    
    # Load data
    df = load_data(train_path, sample_size=sample_size)
    if df is None:
        return None
    
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample data:")
    print(df.head())
    
    # Prepare features and target
    print("\n" + "="*50)
    print("BUILDING FEATURE PIPELINE")
    print("="*50)
    
    # Create feature engineering pipeline
    feature_pipeline = Pipeline([
        ('datetime_features', DatetimeFeatureExtractor('pickup_datetime')),
        ('distance', DistanceCalculator()),
        ('landmarks', LandmarkDistanceCalculator()),
        ('outlier_removal', OutlierRemover()),
        ('feature_selection', FeatureSelector())
    ])
    
    # Separate features and target
    X = df.drop('fare_amount', axis=1, errors='ignore')
    y = df['fare_amount'] if 'fare_amount' in df.columns else None
    
    if y is None:
        print("ERROR: 'fare_amount' column not found in training data!")
        return None
    
    # Remove rows with missing target values
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nData after removing NaN targets: {len(X)} rows")
    
    # Transform features
    print("\nTransforming features...")
    try:
        X_transformed = feature_pipeline.fit_transform(X)
        y_transformed = y.loc[X_transformed.index]
        
        print(f"Transformed features shape: {X_transformed.shape}")
        print(f"Target shape: {y_transformed.shape}")
    except Exception as e:
        print(f"ERROR during feature transformation: {e}")
        return None
    
    # Split data
    print("\nSplitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y_transformed, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Train model
    print("\n" + "="*50)
    print("TRAINING XGBOOST MODEL")
    print("="*50)
    
    model = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training in progress...")
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              verbose=False)
    
    # Evaluate
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    
    train_rmse = np.sqrt(np.mean((y_train - train_predictions) ** 2))
    val_rmse = np.sqrt(np.mean((y_val - val_predictions) ** 2))
    
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Validation R² Score: {val_score:.4f}")
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Validation RMSE: ${val_rmse:.2f}")
    
    # Save pipeline and model
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    
    pipeline_data = {
        'feature_pipeline': feature_pipeline,
        'model': model,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_score,
        'val_r2': val_score,
        'feature_names': feature_pipeline.named_steps['feature_selection'].feature_columns,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_samples': len(X_train)
    }
    
    with open('taxi_fare_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline_data, f)
    
    print("✓ Model saved to taxi_fare_pipeline.pkl")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"\nModel Performance Summary:")
    print(f"  • Validation R²: {val_score:.2%}")
    print(f"  • Validation RMSE: ${val_rmse:.2f}")
    print(f"  • Training samples: {len(X_train):,}")
    print(f"  • Feature count: {X_train.shape[1]}")
    
    return pipeline_data


if __name__ == "__main__":
    print("\n" + "="*50)
    print("NYC TAXI FARE PREDICTION - MODEL TRAINING")
    print("="*50 + "\n")
    
    # Train with 500k samples (adjust based on your dataset size and memory)
    # Set sample_size=None to use the entire dataset
    result = train_model(
        train_path='data/raw/train.csv',
        sample_size=500000  # Use 500k rows for faster training
    )
    
    if result:
        print("\n✓ Training completed successfully!")
        print("  You can now run predict.py to make predictions on test.csv")
        print("  Or run the Streamlit app: streamlit run app.py")
    else:
        print("\n✗ Training failed. Please check the errors above.")
