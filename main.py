from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the trained model
model = joblib.load("iris_model.pkl")

# Define a class for the input data
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict_iris(data: IrisInput):
    # Prepare input for prediction
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    
    # Make prediction
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
