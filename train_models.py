import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import matplotlib.pyplot as plt

# Configure TensorFlow for CPU-only and memory efficiency
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')  # Force CPU-only mode
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Dataset paths
CROP_DATA_PATH = r"D:\Smart-Agriculture\Datasets\Crop Prediction\Crop_recommendation.csv"
DISEASE_ROOT_PATH = r"D:\Smart-Agriculture\Datasets\Plant Disease"
WEATHER_DATA_PATH = r"D:\Smart-Agriculture\Datasets\Weather Prediction"

def train_crop_model():
    """Train a lightweight neural network for crop recommendation"""
    try:
        print("\nLoading crop data...")
        data = pd.read_csv(CROP_DATA_PATH)
        X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = data['label']
        
        # Encode and scale
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42)
        
        # Smaller model architecture
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(len(le.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Training crop model...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[EarlyStopping(patience=5)],
            verbose=1
        )
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model.save('models/crop_model.h5')
        with open('models/crop_preprocessor.pkl', 'wb') as f:
            pickle.dump({'scaler': scaler, 'encoder': le}, f)
            
        print(f"Crop model trained. Validation accuracy: {max(history.history['val_accuracy']):.2f}")
        
    except Exception as e:
        print(f"Error in crop model: {str(e)}")

def train_disease_model():
    """Optimized plant disease detection for low-power hardware"""
    try:
        print("\nPreparing plant disease data...")
        
        # Lightweight data loading
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
            dtype=np.float32
        )
        
        img_size = 160  # Reduced from 300
        batch_size = 8   # Reduced from 32
        
        train_generator = train_datagen.flow_from_directory(
            DISEASE_ROOT_PATH,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            seed=42
        )
        
        val_generator = train_datagen.flow_from_directory(
            DISEASE_ROOT_PATH,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            seed=42
        )
        
        # MobileNetV2 (lightweight)
        base_model = MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            alpha=0.35  # Smaller model
        )
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(len(train_generator.class_indices), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Training disease model (this may take 2-3 hours)...")
        history = model.fit(
            train_generator,
            steps_per_epoch=50,  # Reduced steps
            validation_data=val_generator,
            validation_steps=20,
            epochs=30,
            callbacks=[
                EarlyStopping(patience=5),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
            ],
            verbose=1
        )
        
        # Save model
        model.save('models/disease_model.h5')
        with open('models/class_indices.pkl', 'wb') as f:
            pickle.dump(train_generator.class_indices, f)
            
        print(f"Disease model trained. Validation accuracy: {max(history.history['val_accuracy']):.2f}")
        
    except Exception as e:
        print(f"Error in disease model: {str(e)}")

def train_weather_model1():
    """Simplified weather forecasting model"""
    try:
        print("\nLoading weather data...")
        weather_data = pd.read_csv(WEATHER_DATA_PATH)
        weather_data['date'] = pd.to_datetime(weather_data['date'])
        
        # Simple features
        weather_data['day_of_year'] = weather_data['date'].dt.dayofyear
        weather_data['hour'] = weather_data['date'].dt.hour
        
        # Single-target prediction
        X = weather_data[['day_of_year', 'hour']].values
        y = weather_data['temperature'].values
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y.reshape(-1, 1))
        
        # Simple LSTM model
        model = Sequential([
            tf.keras.layers.LSTM(32, input_shape=(None, 2)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss='mse'
        )
        
        print("Training weather model...")
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        history = model.fit(
            X_reshaped, y_scaled,
            validation_split=0.2,
            epochs=30,
            batch_size=32,
            callbacks=[EarlyStopping(patience=5)],
            verbose=1
        )
        
        # Save model
        model.save('models/weather_model.h5')
        with open('models/weather_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        print("Weather model trained.")
        
    except Exception as e:
        print(f"Error in weather model: {str(e)}")
        
def load_and_preprocess_weather_data():
    """Load and preprocess multiple weather CSV files"""
    base_path = r"D:\Smart-Agriculture\Datasets\Weather Prediction"
    
    # Load all relevant files
    try:
        temperature = pd.read_csv(os.path.join(base_path, "temperature.csv"))
        humidity = pd.read_csv(os.path.join(base_path, "humidity.csv"))
        pressure = pd.read_csv(os.path.join(base_path, "pressure.csv"))
        wind_speed = pd.read_csv(os.path.join(base_path, "wind_speed.csv"))
        wind_direction = pd.read_csv(os.path.join(base_path, "wind_direction.csv"))
        weather_desc = pd.read_csv(os.path.join(base_path, "weather_description.csv"))
        city_attrs = pd.read_csv(os.path.join(base_path, "city_attributes.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Weather data file not found: {str(e)}")

    # Merge all data into a single DataFrame
    dfs = [temperature, humidity, pressure, wind_speed, wind_direction, weather_desc]
    weather_data = dfs[0]
    
    for df in dfs[1:]:
        weather_data = weather_data.merge(df, on=['datetime', 'city_id'])
    
    # Add city attributes
    weather_data = weather_data.merge(city_attrs, on='city_id')
    
    # Convert datetime
    weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
    
    # Handle missing values
    weather_data = weather_data.interpolate(method='linear')
    weather_data = weather_data.fillna(method='bfill').fillna(method='ffill')
    
    return weather_data

def train_weather_model():
    """Train an LSTM neural network for multi-parameter weather forecasting"""
    try:
        # Load and preprocess data
        weather_data = load_and_preprocess_weather_data()
        
        # Focus on one city for simplicity (can extend to multiple cities)
        city_name = "New York"  # Change to your target city
        city_data = weather_data[weather_data['City'] == city_name].copy()
        
        # Create time features
        city_data['day_of_year'] = city_data['datetime'].dt.dayofyear
        city_data['hour'] = city_data['datetime'].dt.hour
        city_data['month'] = city_data['datetime'].dt.month
        
        # Select features and target
        features = [
            'temperature', 'humidity', 'pressure', 
            'wind_speed', 'wind_direction', 'day_of_year',
            'hour', 'month'
        ]
        
        target = 'temperature'  # Can be changed to predict other parameters
        
        # Create sequences for LSTM
        sequence_length = 24 * 7  # 1 week of hourly data
        X, y = [], []
        
        for i in range(len(city_data) - sequence_length):
            X.append(city_data[features].iloc[i:i+sequence_length].values)
            y.append(city_data[target].iloc[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale data
        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
        X_test_scaled = X_scaler.transform(
            X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
        
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
        
        # LSTM model
        model = Sequential([ tf.keras.layers.LSTM(128, input_shape=(sequence_length, len(features))),tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(1) ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_test_scaled, y_test_scaled),
            epochs=50,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
            ],
            verbose=1
        )
        
        # Save model and scalers
        model.save('models/weather_model.h5')
        with open('models/weather_preprocessor.pkl', 'wb') as f:
            pickle.dump({
                'X_scaler': X_scaler,
                'y_scaler': y_scaler,
                'features': features,
                'target': target,
                'sequence_length': sequence_length,
                'city_name': city_name
            }, f)
            
        print(f"\nFinal Test MAE: {model.evaluate(X_test_scaled, y_test_scaled)[1]:.4f}")
        
    except Exception as e:
        print(f"Error training weather model: {str(e)}")

def plot_training_history(history, model_name):
    """Plot training metrics"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training.png')
    plt.close()

if __name__ == '__main__':
    # Clear memory before starting
    import gc
    gc.collect()
    
    print("=== Starting Smart Agriculture Model Training ===")
    print("Note: This will take several hours on a laptop")
    
   # train_crop_model()
  #  disease_history = train_disease_model()
    train_weather_model()
    
   # if disease_history:
   #     plot_training_history(disease_history, "Disease_Model")
    
    print("\n=== All models trained successfully! ===")
    print("Models saved in 'models/' directory")