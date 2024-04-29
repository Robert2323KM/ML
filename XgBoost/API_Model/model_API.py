from flask import Flask
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib
import numpy as np
import requests
from io import BytesIO
import zipfile

app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Car Price Prediction API',
    description='API for predicting car prices based on various features.')

ns = api.namespace('predict', description='Car Price Predictions')

parser = api.parser()
parser.add_argument(
    'url',
    type=str,
    required=True,
    help='URL to the dataset (CSV format expected).',
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.List(fields.Float),
})

models = [joblib.load(f'xgb_model{i}.joblib') for i in range(1, 3)]

@ns.route('/')
class PricePredictionApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        dataset_url = args['url']

        response = requests.get(dataset_url)
        if response.status_code != 200:
            api.abort(500, "Failed to fetch data from provided URL.")

        with zipfile.ZipFile(BytesIO(response.content)) as thezip:
            with thezip.open(thezip.namelist()[0]) as thefile:
                data = pd.read_csv(thefile)

        data['car_age'] = 2023 - data['Year']
        data.drop(['Year'], axis=1, inplace=True)

        predictions = []
        for i, model in enumerate(models, 1):
            model_predictions = model.predict(data)
            predictions.append(model_predictions)
            # Log model use and a preview of its predictions
            print(f"Model {i} used for predictions. Preview of predictions: {model_predictions[:5]}")

        avg_predictions = np.mean(predictions, axis=0).tolist()

        return {'result': avg_predictions}

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
