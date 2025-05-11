from pydantic import BaseModel, Field
from typing import Optional, List

class PredictionInput(BaseModel):
    """
    Input model for the prediction endpoint
    """
    age_at_enrollment: int = Field(
        ...,
        ge=16,
        le=80,
        description="Age of the student at enrollment"
    )

    curricular_units_1st_sem_enrolled: int = Field(
        ...,
        ge=0,
        description="Number of curricular units enrolled in the 1st semester"
    )

    curricular_units_1st_sem_approved: int = Field(
        ...,
        ge=0,
        description="Number of curricular units approved in the 1st semester"
    )

    curricular_units_2nd_sem_enrolled: int = Field(
        ...,
        ge=0,
        description="Number of curricular units enrolled in the 2nd semester"
    )

    curricular_units_2nd_sem_approved: int = Field(
        ...,
        ge=0,
        description="Number of curricular units approved in the 2nd semester"
    )

    unemployment_rate: float = Field(
        ...,
        ge=0,
        le=100,
        description="Unemployment rate at the time of enrollment"
    )

    class Config:
        schema_extra = {
            "example": {
                "age_at_enrollment": 20,
                "curricular_units_1st_sem_enrolled": 6,
                "curricular_units_1st_sem_approved": 5,
                "curricular_units_2nd_sem_enrolled": 6,
                "curricular_units_2nd_sem_approved": 5,
                "unemployment_rate": 10.8
            }
        }

class PredictionOutput(BaseModel):
    """
    Output model for the prediction endpoint
    """
    prediction: str
    dropout_probability: float
    graduate_probability: float

class TrainingResult(BaseModel):
    """
    Training result model
    """
    success: bool
    accuracy: float
    message: str
