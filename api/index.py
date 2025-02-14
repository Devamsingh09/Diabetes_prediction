from fastapi import FastAPI
import numpy as np
import pickle
from Model import Diabetes

app = FastAPI()

# ✅ Load trained model with debugging
try:
    with open("model.pkl", "rb") as file:
        classifier = pickle.load(file)

    if not hasattr(classifier, "predict"):
        raise TypeError("Loaded object is not a model. Check model.pkl!")

except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None  # Prevents crashes


@app.get("/")
def index():
    return {"message": "Hello People!"}


@app.post("/predict")
def predict_diabetes(data: Diabetes):
    if classifier is None:
        return {"error": "Model failed to load. Check model.pkl!"}

    data_dict = data.model_dump()
    input_data = np.array([[
        data_dict['Pregnancies'], data_dict['Glucose'], data_dict['BloodPressure'],
        data_dict['SkinThickness'], data_dict['Insulin'], data_dict['BMI'],
        data_dict['DiabetesPedigreeFunction'], data_dict['Age']
    ]])

    try:
        prediction = classifier.predict(input_data)
        result = "You have diabetes!" if prediction[0] == 1 else "You are safe!"
        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}
