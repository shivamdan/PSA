from fastapi import FastAPI
import uvicorn
from predictor import load_model, make_predictions
import numpy as np

app = FastAPI()

model = load_model('model.pkl')

@app.get('/')
def index_route():
    return {"Health":"Ok"}

@app.post('/predict')
def prediction(temperature, luminosity, radius, abs_mag):
    input_features = [[temperature, luminosity, radius, abs_mag]]
    predicted_class, probabs, classes = make_predictions(model, input_features)
    return {
        'predicted_probabilities' : dict(zip(classes, probabs)),
        'predicted_class' : predicted_class,
        'confidence_score' : str(round(np.max(probabs),3)*100) + '%' # Returning the maximum value among all the probabilities
    }