from flask import Flask, jsonify, request
import pandas as pd
import joblib
import numpy as np
import requests
from io import BytesIO
import zipfile
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

app = Flask(__name__)

# Load the pre-trained models
models = [joblib.load(f'xgb_model{i}.joblib') for i in range(1, 3)]  # Adjust range based on model count

@app.route('/predict', methods=['GET'])
def predict():
    data_url = request.args.get('url')
    if not data_url:
        return jsonify({'error': 'Missing URL parameter'}), 400

    try:
        print(f"Fetching data from: {data_url}")
        response = requests.get(data_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch data'}), 500

        with zipfile.ZipFile(BytesIO(response.content)) as thezip:
            with thezip.open(thezip.namelist()[0]) as thefile:
                data = pd.read_csv(thefile)

        data['car_age'] = 2023 - data['Year']
        data.drop(['Year'], axis=1, inplace=True)

        predictions = []
        for i, model in enumerate(models, 1):
            model_predictions = model.predict(data)
            predictions.append(model_predictions)
            print(f"Model {i} predicted: {model_predictions[:5]}")  # Print the first few predictions of each model

        avg_predictions = np.mean(predictions, axis=0)
        results = {'Predictions': [{'ID': int(idx), 'Predicted_Price': float(pred)} for idx, pred in zip(data.index, avg_predictions)]}
        return jsonify(results), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
