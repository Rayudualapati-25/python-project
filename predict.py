"""
NYC Taxi Fare Prediction - Generate Predictions
This script loads test.csv and generates predictions using the trained model.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os
from sklearn.base import BaseEstimator, TransformerMixin


# Custom Transformer Classes (needed for unpickling the model)
class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract datetime features: year, month, day, weekday, hour"""
    
    def __init__(self, datetime_col='pickup_datetime'):
        self.datetime_col = datetime_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
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
            X[f'{name}_pickup_distance'] = self._haversine_distance(
                lon, lat, X['pickup_longitude'], X['pickup_latitude']
            )
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


def load_model(model_path='taxi_fare_pipeline.pkl'):
    """Load the trained model pipeline"""
    try:
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        print("✓ Model loaded successfully")
        print(f"  • Training date: {pipeline_data.get('training_date', 'Unknown')}")
        print(f"  • Validation RMSE: ${pipeline_data.get('val_rmse', 0):.2f}")
        print(f"  • Validation R²: {pipeline_data.get('val_r2', 0):.2%}")
        
        return pipeline_data
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        print("\nPlease train the model first by running: python train.py")
        return None
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None


def load_test_data(test_path='data/raw/test.csv'):
    """Load test dataset"""
    try:
        print(f"\nLoading test data from {test_path}...")
        df = pd.read_csv(test_path)
        print(f"✓ Loaded {len(df):,} test samples")
        print(f"  Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"ERROR: Test file not found at {test_path}")
        print("\nPlease ensure test.csv is in the data/raw/ directory.")
        print("You can download it from Kaggle:")
        print("https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data")
        return None
    except Exception as e:
        print(f"ERROR loading test data: {e}")
        return None


def make_predictions(test_data, pipeline_data, output_path='data/processed/predictions.csv'):
    """Generate predictions for test data"""
    
    feature_pipeline = pipeline_data['feature_pipeline']
    model = pipeline_data['model']
    
    print("\n" + "="*50)
    print("GENERATING PREDICTIONS")
    print("="*50)
    
    # Store original index/key if exists
    has_key = 'key' in test_data.columns
    if has_key:
        original_keys = test_data['key'].copy()
    
    try:
        # Transform features
        print("\nTransforming test features...")
        X_test_transformed = feature_pipeline.transform(test_data)
        
        print(f"✓ Transformed {len(X_test_transformed):,} samples")
        print(f"  Feature shape: {X_test_transformed.shape}")
        
        # Make predictions
        print("\nGenerating predictions...")
        predictions = model.predict(X_test_transformed)
        
        print(f"✓ Generated {len(predictions):,} predictions")
        print(f"\nPrediction Statistics:")
        print(f"  • Mean fare: ${predictions.mean():.2f}")
        print(f"  • Median fare: ${np.median(predictions):.2f}")
        print(f"  • Min fare: ${predictions.min():.2f}")
        print(f"  • Max fare: ${predictions.max():.2f}")
        print(f"  • Std dev: ${predictions.std():.2f}")
        
        # Create results dataframe
        if has_key:
            # Match predictions with original keys using the transformed index
            results = pd.DataFrame({
                'key': original_keys.loc[X_test_transformed.index],
                'fare_amount': predictions
            })
        else:
            results = pd.DataFrame({
                'fare_amount': predictions
            })
        
        # Save predictions
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to {output_path}")
        
        # Show sample predictions
        print("\nSample Predictions:")
        print(results.head(10))
        
        return results
        
    except Exception as e:
        print(f"\nERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main prediction workflow"""
    print("\n" + "="*50)
    print("NYC TAXI FARE PREDICTION - INFERENCE")
    print("="*50 + "\n")
    
    # Load model
    pipeline_data = load_model()
    if pipeline_data is None:
        return False
    
    # Load test data
    test_data = load_test_data()
    if test_data is None:
        return False
    
    # Generate predictions
    predictions = make_predictions(
        test_data, 
        pipeline_data,
        output_path='data/processed/predictions.csv'
    )
    
    if predictions is not None:
        print("\n" + "="*50)
        print("PREDICTION COMPLETE!")
        print("="*50)
        print("\n✓ All predictions generated successfully!")
        print(f"  • Total predictions: {len(predictions):,}")
        print(f"  • Output file: data/processed/predictions.csv")
        print("\nYou can now:")
        print("  1. Review predictions in data/processed/predictions.csv")
        print("  2. Run the Streamlit app: streamlit run app.py")
        return True
    else:
        print("\n✗ Prediction failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
