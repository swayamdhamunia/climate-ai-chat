import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents file
intents = json.loads(open('data/intents.json').read())

# Load the words, classes, and model
words = pickle.load(open('data/words.pkl.txt', 'rb'))
classes = pickle.load(open('data/classes.pkl.txt', 'rb'))
model = load_model('data/chatbot_model.h5')

# Load trained Random Forest model
rf_model = joblib.load('data/rf_model.pkl')

# Load soil data CSV
data = pd.read_csv('data/soil_data.csv')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    text = message
    cor = []  # Initialize an empty list to store latitude and longitude values

    tokens = nltk.word_tokenize(text)  # Tokenize using NLTK

    # Extract numeric values (latitude, longitude) from the tokenized message
    for item in tokens:
        if isinstance(item, str):
            if item.isdigit():
                continue  # Skip digits, as they're not floats
            else:
                try:
                    # Check if it can be converted to a float (to handle decimal numbers)
                    float_value = float(item)
                    cor.append(float_value)  # Append to the list
                except ValueError:
                    pass  # Skip non-float items

    # Convert the list to a NumPy array
    cor = np.array(cor)

    if np.any(cor > 0):  # If latitude and longitude are found
        latitude = cor[0]
        longitude = cor[1]
        
        # Load your trained model and CSV data
        try:
            rf_model = joblib.load(r'D:\PYTHON\practice\mergedAIproject\rf_model.pkl')
            data = pd.read_csv(r'D:\PYTHON\practice\DROUGHT-PREDICTION-APP-main\data\soil_data.csv')
            row = data[(data['lat'] == latitude) & (data['lon'] == longitude)]
        except Exception as e:
            return jsonify({"response": f"Error loading models or data: {str(e)}"})

        if row.empty:
            return jsonify({"response": "Coordinates not found in the data."})

        # Extract and expand feature values to match the model's expected format
        features = row[['elevation', 'slope1', 'slope2', 'slope3', 'slope4', 'slope5', 
                        'slope6', 'slope7', 'slope8','aspectN', 'aspectE', 'aspectS', 
                        'aspectW', 'aspectUnknown', 'WAT_LAND', 'NVG_LAND', 'URB_LAND', 
                        'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 
                        'CULT_LAND', 'SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']].copy()

        # Expand the SQ columns (example logic, adapt based on how the model expects the SQ features)
        for i in range(1, 8):
            features[f'SQ{i}_1'] = features[f'SQ{i}'] * 0.1  # Example split
            features[f'SQ{i}_2'] = features[f'SQ{i}'] * 0.2  # Example split
            features[f'SQ{i}_3'] = features[f'SQ{i}'] * 0.3  # Example split
            features[f'SQ{i}_4'] = features[f'SQ{i}'] * 0.4  # Example split
            features[f'SQ{i}_6'] = features[f'SQ{i}'] * 0.6  # Example split
            features[f'SQ{i}_7'] = features[f'SQ{i}'] * 0.7  # Example split
            features.drop(columns=[f'SQ{i}'], inplace=True)  # Remove the original SQ column

        # Create DataFrame for the model with correct number of columns
        feature_columns = [
            'elevation', 'slope1', 'slope2', 'slope3', 'slope4', 'slope5', 
            'slope6', 'slope7', 'slope8', 'aspectN', 'aspectE', 'aspectS', 
            'aspectW', 'aspectUnknown', 'WAT_LAND', 'NVG_LAND', 'URB_LAND', 
            'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 'CULT_LAND',
            'SQ1_1', 'SQ1_2', 'SQ1_3', 'SQ1_4', 'SQ1_6', 'SQ1_7',  
            'SQ2_1', 'SQ2_2', 'SQ2_3', 'SQ2_4', 'SQ2_6', 'SQ2_7',  
            'SQ3_1', 'SQ3_2', 'SQ3_3', 'SQ3_4', 'SQ3_6', 'SQ3_7',  
            'SQ4_1', 'SQ4_2', 'SQ4_3', 'SQ4_4', 'SQ4_6', 'SQ4_7',  
            'SQ5_1', 'SQ5_2', 'SQ5_3', 'SQ5_4', 'SQ5_6', 'SQ5_7',  
            'SQ6_1', 'SQ6_2', 'SQ6_3', 'SQ6_6', 'SQ6_7',            
            'SQ7_1', 'SQ7_2', 'SQ7_3', 'SQ7_4', 'SQ7_5', 'SQ7_6', 'SQ7_7'
        ]

        # Ensure the number of columns is correct
        features_df = pd.DataFrame(features.values.reshape(1, -1), columns=feature_columns)

        # Make prediction
        drought_prediction = rf_model.predict(features_df)
        print(f"Drought prediction: {drought_prediction}")

        if drought_prediction > 0:
            response = f"There is rainfall for coordinates ({latitude}, {longitude})"
        else:
            response = f"No rainfall predicted for coordinates ({latitude}, {longitude})"

    else:
        ints = predict_class(message)
        response = get_response(ints, intents)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
