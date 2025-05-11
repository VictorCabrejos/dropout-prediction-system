import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from .api import router as api_router

app = FastAPI(
    title="Student Dropout Prediction System",
    description="Predicts whether a student will dropout or graduate based on academic and socio-economic factors",
    version="1.0.0"
)

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Include API router
app.include_router(api_router)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the home page
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """
    Render the about page
    """
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}
