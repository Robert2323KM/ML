from flask import Flask, request, jsonify
import numpy as np
import re
import json
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the Flask application
app = Flask(__name__)

def load_model_and_tokenizer():
    """Load the pre-trained model, tokenizer, and MultiLabelBinarizer."""
    print("Loading model, tokenizer, and MultiLabelBinarizer...")
    model = load_model('bilstm_model.h5')
    print("Model loaded!")

    with open('tokenizer.json') as f:
        tokenizer = tokenizer_from_json(json.load(f))
    print("Tokenizer loaded!")

    mlb = joblib.load('mlb.pkl')
    print("MultiLabelBinarizer loaded!")

    return model, tokenizer, mlb

def preprocess_text(text):
    """Clean and normalize the input text."""
    text = re.sub(r'\W', ' ', text).lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Load model, tokenizer, and MultiLabelBinarizer
model, tokenizer, mlb = load_model_and_tokenizer()

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    content = request.json
    if 'plot' not in content:
        return jsonify({"error": "Missing plot text"}), 400

    plot = preprocess_text(content['plot'])
    
    # Tokenize and pad the sequence
    seq = tokenizer.texts_to_sequences([plot])
    pad = pad_sequences(seq, maxlen=600)
    
    # Make predictions
    pred = model.predict(pad)
    genres = mlb.inverse_transform(pred > 0.5)
    
    return jsonify({"predicted_genres": genres[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
