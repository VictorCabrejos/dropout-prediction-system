from fastapi import APIRouter, Depends, HTTPException
from ..models.predictor import DropoutPredictor
from ..models.schemas import PredictionInput, PredictionOutput, TrainingResult
import os
import joblib

router = APIRouter(prefix="/api", tags=["prediction"])

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "models", "dropout_predictor.pkl")
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "dataset.csv")

def get_predictor():
    """
    Dependency that provides the DropoutPredictor instance
    """
    try:
        # Check if model exists
        if os.path.exists(MODEL_PATH):
            return DropoutPredictor(model_path=MODEL_PATH)
        else:
            # Initialize a new predictor and train it
            predictor = DropoutPredictor()
            predictor.train(DATASET_PATH)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            predictor.save_model(MODEL_PATH)
            return predictor
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

@router.post("/predict", response_model=PredictionOutput)
def predict(
    input_data: PredictionInput,
    predictor: DropoutPredictor = Depends(get_predictor)
):
    """
    Make a prediction based on the input features
    """
    try:
        # Map input data to features expected by the model
        features_dict = {
            "Age at enrollment": input_data.age_at_enrollment,
            "Curricular units 1st sem (enrolled)": input_data.curricular_units_1st_sem_enrolled,
            "Curricular units 1st sem (approved)": input_data.curricular_units_1st_sem_approved,
            "Curricular units 2nd sem (enrolled)": input_data.curricular_units_2nd_sem_enrolled,
            "Curricular units 2nd sem (approved)": input_data.curricular_units_2nd_sem_approved,
            "Unemployment rate": input_data.unemployment_rate
        }

        # Make prediction
        result = predictor.predict(features_dict)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/train", response_model=TrainingResult)
def train_model():
    """
    Train the model using the dataset
    """
    try:
        # Initialize a new predictor
        predictor = DropoutPredictor()

        # Train the model
        accuracy = predictor.train(DATASET_PATH)

        # Save the model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        predictor.save_model(MODEL_PATH)

        return TrainingResult(
            success=True,
            accuracy=accuracy,
            message=f"Model trained successfully with accuracy: {accuracy:.4f}"
        )
    except Exception as e:
        return TrainingResult(
            success=False,
            accuracy=0.0,
            message=f"Training failed: {str(e)}"
        )
