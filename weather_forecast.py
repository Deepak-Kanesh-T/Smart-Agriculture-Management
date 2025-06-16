import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests
import io
import os
from datetime import datetime, timedelta
import pickle

## --------------------------
## Data Preparation (Updated)
## --------------------------

def download_noaa_data(station_id="USW00094728", year=2023):
    """Download NOAA GSOD data with updated API endpoint"""
    try:
        # Updated NOAA API endpoint
        url = f"https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-summary-of-the-day&stations={station_id}&startDate={year}-01-01&endDate={year}-12-31&format=json"
        
        response = requests.get(url)
        response.raise_for_status()
        
        # Convert JSON to DataFrame
        df = pd.read_json(io.StringIO(response.text))
        
        if df.empty:
            print(f"No data found for station {station_id} in {year}")
            return None
            
        return df
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def preprocess_weather_data(df):
    """Clean and prepare weather data"""
    # Convert date and set as index
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    
    # Select and rename columns
    features = {
        'TEMP': 'temperature',
        'DEWP': 'dew_point',
        'SLP': 'pressure',
        'WDSP': 'wind_speed',
        'PRCP': 'precipitation',
        'RH': 'humidity'
    }
    
    # Only keep columns that exist in the data
    available_features = {k: v for k, v in features.items() if k in df.columns}
    df = df[available_features.keys()].rename(columns=available_features)
    
    # Convert temperatures from Fahrenheit to Celsius if they exist
    if 'temperature' in df.columns:
        df['temperature'] = (df['temperature'] - 32) * 5/9
    if 'dew_point' in df.columns:
        df['dew_point'] = (df['dew_point'] - 32) * 5/9
    
    # Handle missing values
    df = df.interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
    
    # Add temporal features
    df['day_of_year'] = df.index.dayofyear
    df['day_sin'] = np.sin(df['day_of_year'] * (2 * np.pi / 365))
    df['day_cos'] = np.cos(df['day_of_year'] * (2 * np.pi / 365))
    
    return df

## --------------------------
## Alternative Data Sources
## --------------------------

def get_sample_data():
    """Fallback sample data if API fails"""
    print("Using sample data as fallback...")
    dates = pd.date_range(start="2020-01-01", end="2022-12-31")
    data = {
        'temperature': np.sin(np.arange(len(dates)) * 15 + 20),
        'humidity': np.cos(np.arange(len(dates)) * 20 + 60),
        'pressure': np.random.normal(1015, 5, len(dates)),
        'wind_speed': np.abs(np.random.normal(10, 3, len(dates)))
    }
    return pd.DataFrame(data, index=dates)

## --------------------------
## Model Building (Same as Before)
## --------------------------

def create_sequences(data, target_col, seq_length=7, step=1):
    """Create time series sequences for LSTM"""
    X, y = [], []
    for i in range(0, len(data) - seq_length, step):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length][target_col])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build LSTM model for weather forecasting"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

## --------------------------
## Updated Training Pipeline
## --------------------------

def train_weather_model(use_sample_data=False):
    """Complete training pipeline with fallback"""
    if use_sample_data:
        full_data = get_sample_data()
    else:
        dfs = []
        for year in [2020, 2021, 2022]:
            df = download_noaa_data(year=year)
            if df is not None:
                processed = preprocess_weather_data(df)
                if not processed.empty:
                    dfs.append(processed)
        
        if not dfs:
            print("Falling back to sample data")
            return train_weather_model(use_sample_data=True)
        
        full_data = pd.concat(dfs)
    
    # Rest of the training pipeline remains the same...
    target_col = 'temperature'
    available_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'day_sin', 'day_cos']
    features = [f for f in available_features if f in full_data.columns]
    
    if not features:
        raise ValueError("No valid features found in the data")
    
    data = full_data[features]
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    
    # Create sequences
    seq_length = 14
    X, y = create_sequences(scaled_data, target_col, seq_length)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Build and train model
    model = build_lstm_model((seq_length, len(features)))
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
        ],
        verbose=1
    )
    
    # Save artifacts
    os.makedirs('weather_model', exist_ok=True)
    model.save('weather_model/lstm_model.h5')
    with open('weather_model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('weather_model/features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    return model, scaler, features

## --------------------------
## Main Execution
## --------------------------

if __name__ == '__main__':
    try:
        print("Training weather model...")
        model, scaler, features = train_weather_model()
        
        # Generate sample forecast
        if os.path.exists('weather_model/lstm_model.h5'):
            print("\nModel trained successfully! Generating sample forecast...")
            
            # Create synthetic recent data for demonstration
            dates = pd.date_range(end=datetime.now(), periods=14)
            recent_data = pd.DataFrame({
                'temperature': np.sin(np.arange(len(dates))) * 10 + 20,
                'humidity': np.cos(np.arange(len(dates))) * 15 + 60,
                'pressure': np.random.normal(1015, 5, len(dates)),
                'wind_speed': np.abs(np.random.normal(10, 3, len(dates)))
            }, index=dates)
            
            # Add temporal features
            recent_data['day_of_year'] = recent_data.index.dayofyear
            recent_data['day_sin'] = np.sin(recent_data['day_of_year'] * (2 * np.pi / 365))
            recent_data['day_cos'] = np.cos(recent_data['day_of_year'] * (2 * np.pi / 365))
            
            forecast = make_forecast(model, scaler, features, recent_data, days=7)
            
            print("\nSample 7-Day Forecast:")
            print(forecast)
            
            # Plot results
            plt.figure(figsize=(12, 6))
            plt.plot(recent_data['temperature'], label='Historical')
            plt.plot(forecast, label='Forecast', marker='o')
            plt.title('7-Day Temperature Forecast')
            plt.ylabel('Temperature (°C)')
            plt.xlabel('Date')
            plt.legend()
            plt.grid(True)
            plt.savefig('weather_forecast.png')
            plt.show()
        else:
            print("Model training failed. Please check the error messages.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")