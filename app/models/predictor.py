import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DropoutPredictor:
    def __init__(self, model_path=None):
        """
        Initialize the DropoutPredictor class.

        Args:
            model_path (str): Path to a pre-trained model. If None, a new model will be trained.
        """
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.features = [
            'Age at enrollment',
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (approved)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (approved)',
            'Unemployment rate'
        ]

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def preprocess_data(self, data):
        """
        Preprocess the data for training or prediction.

        Args:
            data (pd.DataFrame): The data to preprocess

        Returns:
            pd.DataFrame: The preprocessed data
        """
        # Select only the features we need
        X = data[self.features]

        return X

    def train(self, dataset_path):
        """
        Train the model using the dataset.

        Args:
            dataset_path (str): Path to the dataset.

        Returns:
            float: The accuracy score of the model
        """
        # Load dataset
        data = pd.read_csv(dataset_path)

        # Prepare features and target
        X = self.preprocess_data(data)
        y = data['Target']

        # Remove rows where the target is not Dropout or Graduate
        mask = y.isin(['Dropout', 'Graduate'])
        X = X[mask]
        y = y[mask]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        accuracy = self.model.score(X_test_scaled, y_test)

        return accuracy

    def predict(self, features_dict):
        """
        Make a prediction using the trained model.

        Args:
            features_dict (dict): Dictionary containing the features for prediction

        Returns:
            dict: Prediction result and probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Create a DataFrame from the input features
        input_df = pd.DataFrame([features_dict])

        # Ensure the DataFrame has all required features
        for feature in self.features:
            if feature not in input_df.columns:
                input_df[feature] = 0

        # Select only the required features in the correct order
        input_df = input_df[self.features]

        # Scale the input features
        input_scaled = self.scaler.transform(input_df)

        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]

        # Get the probability of the predicted class
        if prediction == "Dropout":
            dropout_prob = probabilities[list(self.model.classes_).index("Dropout")]
            graduate_prob = probabilities[list(self.model.classes_).index("Graduate")]
        else:
            dropout_prob = probabilities[list(self.model.classes_).index("Dropout")]
            graduate_prob = probabilities[list(self.model.classes_).index("Graduate")]

        return {
            "prediction": prediction,
            "dropout_probability": float(dropout_prob),
            "graduate_probability": float(graduate_prob)
        }

    def save_model(self, model_path):
        """
        Save the trained model to disk.

        Args:
            model_path (str): Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")

        # Save the model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features
        }, model_path)

        self.model_path = model_path

    def load_model(self, model_path):
        """
        Load a trained model from disk.

        Args:
            model_path (str): Path to the saved model
        """
        # Load the model and scaler
        loaded_data = joblib.load(model_path)

        self.model = loaded_data['model']
        self.scaler = loaded_data['scaler']
        self.features = loaded_data['features']
        self.model_path = model_path
