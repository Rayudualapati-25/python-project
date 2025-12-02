"""
NYC Taxi Fare Prediction - Generate Predictions
This script loads test.csv and generates predictions using the trained model.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os


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
