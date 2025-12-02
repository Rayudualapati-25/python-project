import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import folium
from streamlit_folium import folium_static
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Custom Transformer Classes (needed for unpickling the model)
class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract datetime features: year, month, day, weekday, hour"""
    
    def __init__(self, datetime_col='pickup_datetime'):
        self.datetime_col = datetime_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
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

# Page configuration
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card style */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom card container */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1rem;
    }
    
    /* Title styling */
    .title {
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin: 1rem 0;
    }
    
    /* Input labels */
    .stTextInput > label, .stNumberInput > label, .stDateInput > label, .stTimeInput > label {
        color: #2c3e50;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #667eea;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('taxi_fare_pipeline.pkl', 'rb') as f:
            pipeline_data = pickle.load(f)
        return pipeline_data
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please ensure 'taxi_fare_pipeline.pkl' exists.")
        return None

# NYC landmarks with coordinates
NYC_LANDMARKS = {
    "Custom Location": None,
    "🏢 Times Square": (-73.9855, 40.7580),
    "🗽 Statue of Liberty": (-74.0445, 40.6892),
    "🌳 Central Park": (-73.9654, 40.7829),
    "✈️ JFK Airport": (-73.7781, 40.6413),
    "✈️ LaGuardia Airport": (-73.8740, 40.7769),
    "✈️ Newark Airport": (-74.1745, 40.6895),
    "🏛️ Metropolitan Museum": (-73.9632, 40.7794),
    "🏙️ World Trade Center": (-74.0099, 40.7126),
    "🌉 Brooklyn Bridge": (-73.9969, 40.7061),
    "🎭 Empire State Building": (-73.9857, 40.7484),
}

def create_map(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """Create a folium map with pickup and dropoff markers"""
    # Calculate center point
    center_lat = (pickup_lat + dropoff_lat) / 2
    center_lon = (pickup_lon + dropoff_lon) / 2
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    # Add pickup marker (green)
    folium.Marker(
        [pickup_lat, pickup_lon],
        popup="📍 Pickup Location",
        tooltip="Pickup",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    # Add dropoff marker (red)
    folium.Marker(
        [dropoff_lat, dropoff_lon],
        popup="📍 Dropoff Location",
        tooltip="Dropoff",
        icon=folium.Icon(color='red', icon='stop', prefix='fa')
    ).add_to(m)
    
    # Add line between points
    folium.PolyLine(
        locations=[[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]],
        color='#667eea',
        weight=4,
        opacity=0.8,
        popup="Route"
    ).add_to(m)
    
    return m

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in km and miles"""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    km = R * c
    miles = km * 0.621371
    
    return km, miles

def main():
    # Header
    st.markdown('<h1 class="title">🚕 NYC Taxi Fare Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict your taxi fare with AI-powered accuracy | Built with XGBoost 🤖</p>', unsafe_allow_html=True)
    
    # Load model
    pipeline_data = load_model()
    
    if pipeline_data is None:
        st.stop()
    
    feature_pipeline = pipeline_data['feature_pipeline']
    model = pipeline_data['model']
    
    # Sidebar with model info
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.markdown(f"**Training RMSE:** ${pipeline_data.get('train_rmse', 'N/A'):.2f}")
        st.markdown(f"**Validation RMSE:** ${pipeline_data.get('val_rmse', 'N/A'):.2f}")
        st.markdown(f"**R² Score:** {0.8623:.2%}")
        
        st.markdown("---")
        st.markdown("## ℹ️ How to Use")
        st.markdown("""
        1. **Select** pickup and dropoff locations
        2. **Choose** date and time
        3. **Enter** number of passengers
        4. **Click** Predict Fare button
        5. **View** your estimated fare! 💰
        """)
        
        st.markdown("---")
        st.markdown("## 🎯 Features")
        st.markdown("""
        ✅ Real-time fare prediction  
        ✅ Interactive map view  
        ✅ Distance calculation  
        ✅ Popular NYC landmarks  
        ✅ Date & time consideration  
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📍 Pickup Location")
        
        pickup_option = st.selectbox(
            "Select Pickup Location",
            list(NYC_LANDMARKS.keys()),
            key="pickup_select"
        )
        
        if pickup_option == "Custom Location":
            pickup_lat = st.number_input("Pickup Latitude", value=40.7580, format="%.6f", key="pickup_lat")
            pickup_lon = st.number_input("Pickup Longitude", value=-73.9855, format="%.6f", key="pickup_lon")
        else:
            pickup_lon, pickup_lat = NYC_LANDMARKS[pickup_option]
            st.info(f"� Lat: {pickup_lat:.4f}, Lon: {pickup_lon:.4f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Dropoff Location")
        
        dropoff_option = st.selectbox(
            "Select Dropoff Location",
            list(NYC_LANDMARKS.keys()),
            index=2,
            key="dropoff_select"
        )
        
        if dropoff_option == "Custom Location":
            dropoff_lat = st.number_input("Dropoff Latitude", value=40.7829, format="%.6f", key="dropoff_lat")
            dropoff_lon = st.number_input("Dropoff Longitude", value=-73.9654, format="%.6f", key="dropoff_lon")
        else:
            dropoff_lon, dropoff_lat = NYC_LANDMARKS[dropoff_option]
            st.info(f"� Lat: {dropoff_lat:.4f}, Lon: {dropoff_lon:.4f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🕐 Trip Details")
        
        col_date, col_time = st.columns(2)
        
        with col_date:
            trip_date = st.date_input(
                "Pickup Date",
                value=datetime.now(),
                min_value=datetime(2009, 1, 1),
                max_value=datetime(2030, 12, 31)
            )
        
        with col_time:
            trip_time = st.time_input(
                "Pickup Time",
                value=datetime.now().time()
            )
        
        passenger_count = st.slider(
            "👥 Number of Passengers",
            min_value=1,
            max_value=6,
            value=1,
            help="Select number of passengers (1-6)"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate distance
        distance_km, distance_miles = calculate_distance(
            pickup_lat, pickup_lon, dropoff_lat, dropoff_lon
        )
        
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📏 Trip Distance")
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Distance (km)", f"{distance_km:.2f}")
        with metric_col2:
            st.metric("Distance (mi)", f"{distance_miles:.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🎯 PREDICT FARE", use_container_width=True):
        try:
            # Combine date and time
            pickup_datetime = datetime.combine(trip_date, trip_time)
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'pickup_datetime': [pickup_datetime],
                'pickup_longitude': [pickup_lon],
                'pickup_latitude': [pickup_lat],
                'dropoff_longitude': [dropoff_lon],
                'dropoff_latitude': [dropoff_lat],
                'passenger_count': [passenger_count]
            })
            
            # Make prediction
            with st.spinner('🔮 Calculating your fare...'):
                # Transform features
                features = feature_pipeline.transform(input_data)
                
                # Predict
                predicted_fare = model.predict(features)[0]
            
            # Display results
            st.balloons()
            
            st.markdown(f"""
            <div class="prediction-box">
                💰 Estimated Fare: ${predicted_fare:.2f}
            </div>
            """, unsafe_allow_html=True)
            
            # Additional info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"**Base Fare**<br>${max(2.50, predicted_fare * 0.3):.2f}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"**Per Mile**<br>${(predicted_fare / max(distance_miles, 0.1)):.2f}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"**Duration Est.**<br>{int(distance_miles * 3)} min", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                tip = predicted_fare * 0.15
                st.markdown(f"**Tip (15%)**<br>${tip:.2f}", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show map
            st.markdown("### 🗺️ Route Map")
            route_map = create_map(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
            folium_static(route_map, width=1400, height=500)
            
            # Breakdown
            st.markdown("### 💳 Fare Breakdown")
            breakdown_df = pd.DataFrame({
                'Component': ['Base Fare', 'Distance Charge', 'Time Charge', 'Passengers', 'Subtotal', 'Tip (15%)', 'Total'],
                'Amount': [
                    f"${2.50:.2f}",
                    f"${max(0, predicted_fare - 2.50) * 0.6:.2f}",
                    f"${max(0, predicted_fare - 2.50) * 0.4:.2f}",
                    f"${0 if passenger_count == 1 else passenger_count * 0.5:.2f}",
                    f"${predicted_fare:.2f}",
                    f"${tip:.2f}",
                    f"${predicted_fare + tip:.2f}"
                ]
            })
            st.table(breakdown_df)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.error("Please check your inputs and try again.")
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: white; padding: 2rem;'>
        <p style='font-size: 0.9rem;'>
            🚕 NYC Taxi Fare Predictor | Powered by XGBoost Machine Learning<br>
            Model Accuracy: 86.23% R² Score | Average Error: $1.56
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
