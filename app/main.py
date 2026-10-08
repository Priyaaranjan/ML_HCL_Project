from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Mushroom Edibility Prediction API",
    description="Educational ML API for predicting mushroom edibility",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD FINAL ML PIPELINE
# ============================================================

MODEL_PATH = "model/final_mushroom_pipeline.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# INPUT DATA MODEL
# ============================================================

class MushroomData(BaseModel):

    cap_shape: str
    cap_surface: str
    cap_color: str
    bruises: str
    odor: str
    gill_attachment: str
    gill_spacing: str
    gill_size: str
    gill_color: str
    stalk_shape: str
    stalk_root: str
    stalk_surface_above_ring: str
    stalk_surface_below_ring: str
    stalk_color_above_ring: str
    stalk_color_below_ring: str
    veil_type: str
    veil_color: str
    ring_number: str
    ring_type: str
    spore_print_color: str
    population: str
    habitat: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Mushroom Edibility Prediction API",
        "status": "running",
        "model": "Random Forest + Chi-Square Feature Selection"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "Final Mushroom Edibility Pipeline",
        "message": "API is ready for predictions"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: MushroomData):

    # Convert input to DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    # Get class names
    class_names = model.classes_

    # Find probability of predicted class
    prediction_index = list(class_names).index(prediction)

    confidence = probabilities[prediction_index] * 100

    # Convert class code to readable result
    if prediction == "e":
        result = "Edible"
    else:
        result = "Poisonous"

    # Return result
    return {
        "prediction": result,
        "class_code": prediction,
        "confidence": round(confidence, 2),
        "message": (
            "Educational prediction only. "
            "Do not use this result to determine whether "
            "a real mushroom is safe to eat."
        )
    }