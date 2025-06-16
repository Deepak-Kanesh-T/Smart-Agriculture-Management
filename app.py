from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
from prophet import Prophet
import pandas as pd
import chromadb
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os
from datetime import datetime
import requests

app = Flask(__name__)
app.secret_key = '12345_don'

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/smart-agri"
mongo = PyMongo(app)

# ChromaDB configuration
chroma_client = chromadb.Client()
disease_collection = chroma_client.create_collection(name="disease_embeddings")

# Load ML models
with open('models/random_forest_classifier.pkl', 'rb') as f:
    crop_model = pickle.load(f)
#crop_model = load_model('models/crop_model.h5')
disease_model = load_model('models/disease_model.h5')

with open('weather_model/scaler.pkl', 'rb') as f:
    weather_model = pickle.load(f)

# Disease classes
disease_classes = ["Potato___Late_blight", "Tomato__Target_Spot", "Tomato__Tomato_mosaic_virus", "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato_Bacterial_spot", "Tomato_Early_blight","Tomato_healthy",
"Tomato_Late_blight",
"Tomato_Leaf_Mold",
"Tomato_Septoria_leaf_spot",
"Tomato_Spider_mites_Two_spotted_spider_mite",
"Pepper__bell___Bacterial_spot",
"Pepper__bell___healthy",
"Potato___Early_blight",
"Potato___healthy"]

# OpenWeatherMap API key
OWM_API_KEY = 'http://api.weatherapi.com/v1/current.json?key=5bd197a93e954a69bf634259251606&q=10.9974, 76.9589&days=5'

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = mongo.db.users.find_one({'username': username})
        
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if mongo.db.users.find_one({'username': username}):
            flash('Username already exists')
        else:
            hashed_password = generate_password_hash(password)
            mongo.db.users.insert_one({
                'username': username,
                'password': hashed_password
            })
            flash('Registration successful. Please login.')
            return redirect(url_for('login'))
    
    
    return render_template('login.html', register=True)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', username=session['username'])

@app.route('/weather', methods=['GET', 'POST'])
def weather_forecast():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    forecast_data = None
    if request.method == 'POST':
        location = request.form['location']
        days = int(request.form['days'])
        
        # Get current weather
        current_weather = requests.get(
            f'http://api.weatherapi.com/v1/current.json?key=5bd197a93e954a69bf634259251606&q={10.9974, 76.9589}9&days=4'
        ).json()
        
        # Generate future dates
        # future = weather_model.make_future_dataframe(periods=days)
        # forecast = weather_model.predict(future)
        
        # Prepare forecast data
        forecast_data = {
            'current': {
                'temp': current_weather['main']['temp'],
                'humidity': current_weather['main']['humidity'],
                'description': current_weather['weather'][0]['description'],
                'icon': current_weather['weather'][0]['icon']
            },
            # 'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days).to_dict('records')
        }
    
    return render_template('weather.html', forecast=forecast_data)

"""@app.route('/weather', methods=['GET', 'POST'])
def weather_forecast():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    forecast_data = None
    if request.method == 'POST':
        location = request.form['location']
        days = int(request.form['days'])
        
        try:
            # Get current weather from API
            current_weather = requests.get(
                #f'http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OWM_API_KEY}&units=metric'
                f'http://api.weatherapi.com/v1/current.json?key=5bd197a93e954a69bf634259251606&q={10.9974, 76.9589}9&days=4'
            ).json()
            
            # Load model and preprocessors
            weather_preprocessor = None
            with open('weather_model/scaler.pkl', 'rb') as f:
                weather_preprocessor = pickle.load(f)
            
            model = load_model('weather_model/lstm_model.h5')
            
            # Get historical data for the selected city
            weather_data = load_and_preprocess_weather_data()
            city_data = weather_data[weather_data['City'] == weather_preprocessor['city_name']].copy()
            
            # Prepare the most recent sequence
            last_sequence = city_data[weather_preprocessor['features']].iloc[-weather_preprocessor['sequence_length']:].values
            last_sequence_scaled = weather_preprocessor['X_scaler'].transform(last_sequence)
            
            # Generate forecast
            forecast = []
            current_input = last_sequence_scaled.reshape(1, weather_preprocessor['sequence_length'], len(weather_preprocessor['features']))
            
            for _ in range(days * 24):  # Daily forecast (can adjust for hourly)
                # Predict next step
                pred = model.predict(current_input)
                pred_value = weather_preprocessor['y_scaler'].inverse_transform(pred)[0][0]
                forecast.append(pred_value)
                
                # Update input sequence
                new_row = np.zeros_like(current_input[0, 0, :])
                new_row[0] = pred_value  # Update temperature
                # For other features, we could use persistence or simple models
                # Here we just keep them constant for simplicity
                for i in range(1, len(weather_preprocessor['features'])):
                    new_row[i] = current_input[0, -1, i]
                
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, :] = new_row
            
            # Prepare forecast data for display
            forecast_dates = pd.date_range(
                start=pd.Timestamp.now(),
                periods=days,
                freq='D'
            )
            
            # Aggregate hourly forecasts to daily
            daily_forecast = []
            for i in range(days):
                day_values = forecast[i*24:(i+1)*24]
                daily_forecast.append({
                    'date': forecast_dates[i].strftime('%Y-%m-%d'),
                    'avg_temp': np.mean(day_values),
                    'min_temp': np.min(day_values),
                    'max_temp': np.max(day_values)
                })
            
            forecast_data = {
                'current': {
                    'temp': current_weather['main']['temp'],
                    'humidity': current_weather['main']['humidity'],
                    'pressure': current_weather['main']['pressure'],
                    'wind_speed': current_weather['wind']['speed'],
                    'description': current_weather['weather'][0]['description'],
                    'icon': current_weather['weather'][0]['icon']
                },
                'forecast': daily_forecast,
                'city': weather_preprocessor['city_name']
            }
            
        except Exception as e:
            flash(f'Error generating forecast: {str(e)}')
    
    return render_template('weather.html', forecast=forecast_data)

# List of crops corresponding to model output"""
crop_labels = [
    "apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton", 
    "grapes", "jute", "kidneybeans", "lentil", "maize", "mango", "mothbeans", 
    "mungbean", "muskmelon", "orange", "papaya", "pigeonpeas", "pomegranate", 
    "rice", "watermelon"
]

@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if 'username' not in session:
        return redirect(url_for('login'))

    recommendation = None
    if request.method == 'POST':
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        prediction = crop_model.predict(input_data)  # prediction is array of probabilities

        predicted_index = np.argmax(prediction)
        recommendation = crop_labels[predicted_index]  # Convert index to crop name

    return render_template('crop.html', recommendation=recommendation)

    
  
@app.route('/disease-detection', methods=['GET', 'POST'])
def disease_detection():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    result = None
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image uploaded')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No image selected')
            return redirect(request.url)
        
        if file:
            # Save the image temporarily
            img = Image.open(io.BytesIO(file.read()))
            img = img.resize((160, 160))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # Make prediction
            predictions = disease_model.predict(img_array)
            predicted_class = disease_classes[np.argmax(predictions[0])]
            confidence = round(np.max(predictions[0]) * 100, 2)
            
            # Store embedding in ChromaDB
            embedding = predictions[0].tolist()
            disease_collection.add(
                embeddings=[embedding],
                metadatas=[{"label": predicted_class, "confidence": confidence}],
                ids=[str(datetime.now().timestamp())]
            )
            
            result = {
                'class': predicted_class,
                'confidence': confidence,
                'image': file.filename
            }
    
    return render_template('disease.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)