import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from prophet import Prophet
import os
import shutil
from PIL import Image

# Dataset paths
CROP_DATA_PATH = r"D:\Smart-Agriculture\Datasets\Crop Prediction\Crop_recommendation.csv"
DISEASE_ROOT_PATH = r"D:\Smart-Agriculture\Datasets\Plant Disease"
WEATHER_DATA_PATH = r"D:\Smart-Agriculture\Datasets\Weather Prediction\historical_weather.csv"

def train_crop_model():
    """Train a neural network for crop recommendation"""
    try:
        # Load and preprocess data
        data = pd.read_csv(CROP_DATA_PATH)
        X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = data['label']
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42)
        
        # Neural Network model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(len(le.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nCrop recommendation model accuracy: {test_acc:.4f}")
        
        # Save model and preprocessing objects
        os.makedirs('models', exist_ok=True)
        model.save('models/crop_model.h5')
        with open('models/crop_preprocessor.pkl', 'wb') as f:
            pickle.dump({'scaler': scaler, 'encoder': le}, f)
            
    except Exception as e:
        print(f"Error training crop model: {str(e)}")

def train_disease_model():
    """Train a high-accuracy CNN for plant disease detection using EfficientNetB3"""
    try:
        # Get all plant-disease folders
        disease_folders = [f for f in os.listdir(DISEASE_ROOT_PATH) 
                         if os.path.isdir(os.path.join(DISEASE_ROOT_PATH, f)) and not f.startswith('.')]
        
        # Create train/valid directories
        shutil.rmtree('data', ignore_errors=True)  # Clear previous data
        os.makedirs('data/train', exist_ok=True)
        os.makedirs('data/val', exist_ok=True)
        
        # Create organized dataset structure with stratified split
        for folder in disease_folders:
            class_path = os.path.join(DISEASE_ROOT_PATH, folder)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Ensure minimum 2 samples per class for validation
            if len(images) < 5:
                print(f"Warning: Class {folder} has only {len(images)} images")
                continue
                
            train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
            
            # Create directories and copy files
            for split, imgs in [('train', train_images), ('val', val_images)]:
                dest_dir = os.path.join('data', split, folder)
                os.makedirs(dest_dir, exist_ok=True)
                for img in imgs:
                    shutil.copy2(os.path.join(class_path, img), 
                                os.path.join(dest_dir, img))
        
        # Enhanced data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            channel_shift_range=50
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Increased image size for B3
        img_size = 300
        batch_size = 32
        
        train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            'data/val',
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Save class indices
        class_indices = train_generator.class_indices
        with open('models/class_indices.pkl', 'wb') as f:
            pickle.dump(class_indices, f)
        
        # Enhanced EfficientNetB3 model
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size, img_size, 3),
            drop_connect_rate=0.4
        )
        
        # Freeze first 150 layers
        for layer in base_model.layers[:150]:
            layer.trainable = False
            
        # Custom head with regularization
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(class_indices), activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Custom learning rate schedule
        initial_lr = 0.001
        def lr_scheduler(epoch):
            if epoch < 10:
                return initial_lr
            elif epoch < 20:
                return initial_lr * 0.1
            else:
                return initial_lr * 0.01
                
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5),
            ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True),
            tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]
        
        # Compile with additional metrics
        model.compile(
            optimizer=Adam(learning_rate=initial_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                   tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name='recall')]
        )
        
        # Train model
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch_size,
            epochs=50,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save('models/disease_model.h5')
        
        # Plot training history
        plot_training_history(history)
        
        return history
        
    except Exception as e:
        print(f"Error training disease model: {str(e)}")
        raise

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

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

if __name__ == '__main__':
    print("Training crop recommendation model...")
    train_crop_model()
    
    print("\nTraining plant disease detection model...")
    train_disease_model()
    
    print("\nTraining weather forecasting model...")
    train_weather_model()
    
    print("\nAll models trained and saved successfully!")