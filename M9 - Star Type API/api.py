from fastapi import FastAPI
from star_data import StarProperties, StarTypePrediction
import uvicorn
from predictor import load_model, make_predictions
import numpy as np

app = FastAPI()

model = load_model('model.pkl')

@app.get('/')
def index_route():
    return {"Health":"Ok"}

@app.post('/predict', response_model=StarTypePrediction)
def prediction(sp: StarProperties):
# def prediction(temperature, luminosity, radius, abs_mag):
    input_features = [[sp.temperature, sp.luminosity, sp.radius, sp.abs_mag]]
    predicted_class, probabs, classes = make_predictions(model, input_features)
    pred_probs = dict(zip(classes, probabs))
    sorted_pred_probs = dict(sorted(pred_probs.items(), key = lambda item:item[1], reverse=True))
    return {
        'predicted_probabilities' : sorted_pred_probs,
        'predicted_class' : predicted_class,
        'confidence_score' : str(round(np.max(probabs),3)*100) + '%' # Returning the maximum value among all the probabilities
    }

'''
Test on real star data taken from Wikipedia
Betelguese (Supergiant) : https://en.wikipedia.org/wiki/Betelgeuse
Beta Pictoris (Main Sequence) : https://en.wikipedia.org/wiki/Beta_Pictoris
Sirius (White Dwarf) : https://en.wikipedia.org/wiki/Sirius
'''