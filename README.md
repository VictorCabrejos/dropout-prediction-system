# Student Dropout Prediction System

A machine learning-based web application to predict whether a student is likely to drop out or graduate based on academic performance and socio-economic factors.

## Features

- Predict student dropout probability using machine learning
- User-friendly web interface
- RESTful API for integration with other systems
- Interactive data visualization

## Technology Stack

- **Backend**: FastAPI
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Data Visualization**: Chart.js

## Project Structure

```
dropout-prediction-system/
│
├── app/
│   ├── api/                 # API endpoints
│   ├── models/              # ML models and Pydantic schemas
│   ├── static/              # Static assets (CSS, JS)
│   ├── templates/           # HTML templates
│   └── main.py              # FastAPI app
│
├── models/                  # Saved ML model files
├── notebooks/               # Jupyter notebooks for exploration
├── dataset.csv              # Dataset for training
├── requirements.txt         # Dependencies
└── run.py                   # Startup script
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python run.py
```

The application will be available at http://localhost:8000

## API Documentation

The API documentation is automatically generated and available at http://localhost:8000/docs when running the application.

### Prediction API

```
POST /api/predict
```

**Request Body**:

```json
{
  "age_at_enrollment": 20,
  "curricular_units_1st_sem_enrolled": 6,
  "curricular_units_1st_sem_approved": 5,
  "curricular_units_2nd_sem_enrolled": 6,
  "curricular_units_2nd_sem_approved": 5,
  "unemployment_rate": 10.8
}
```

**Response**:

```json
{
  "prediction": "Graduate",
  "dropout_probability": 0.27,
  "graduate_probability": 0.73
}
```

## Model Training

The model is trained using logistic regression on the provided dataset. The features used for prediction include:

- Age at enrollment
- Curricular units enrolled and approved in 1st semester
- Curricular units enrolled and approved in 2nd semester
- Unemployment rate

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
