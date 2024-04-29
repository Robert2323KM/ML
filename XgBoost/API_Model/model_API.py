from flask import Flask
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib
import requests
from io import BytesIO
import zipfile

# Create Flask application
app = Flask(__name__)

# Define API using Flask-RESTx
api = Api(
    app,
    version='1.0',
    title='Car Price Prediction API',
    description='API for predicting car prices based on various features.')

# Namespace for predictions
ns = api.namespace('predict', description='Car Price Predictions')

# Parser for incoming request arguments
parser = api.parser()
parser.add_argument(
    'url',
    type=str,
    required=True,
    help='URL to the dataset (CSV format expected).',
    location='args')

# Model for output
resource_fields = api.model('Resource', {
    'result': fields.List(fields.Float),
})

# Load pre-trained XGBoost models
models = [joblib.load(f'xgb_model{i}.joblib') for i in range(1, 17)]

# Define the class for the prediction resource
@ns.route('/')
class PricePredictionApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        dataset_url = args['url']

        # Fetch and prepare the data
        response = requests.get(dataset_url)
        if response.status_code != 200:
            api.abort(500, "Failed to fetch data from provided URL.")

        with zipfile.ZipFile(BytesIO(response.content)) as thezip:
            with thezip.open(thezip.namelist()[0]) as thefile:
                data = pd.read_csv(thefile)

        # Feature engineering
        data['car_age'] = 2023 - data['Year']
        data.drop(['Year'], axis=1, inplace=True)

        # Predict using all models and average the results
        predictions = [model.predict(data) for model in models]
        avg_predictions = np.mean(predictions, axis=0).tolist()  # Convert to list for JSON serialization

        return {'result': avg_predictions}

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
