from fastapi import APIRouter
from .prediction import router as prediction_router

router = APIRouter()
router.include_router(prediction_router)

__all__ = ['router']
