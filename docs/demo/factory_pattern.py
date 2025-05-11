"""
Patrón Factory - Demostración

Este archivo muestra la implementación del patrón Factory para crear diferentes
tipos de modelos de predicción de deserción estudiantil.
"""

# -------------------------------------------------------------------------
# CÓDIGO ESPAGUETI ORIGINAL (SIN PATRÓN FACTORY)
# -------------------------------------------------------------------------
"""
# Sin aplicar el patrón Factory, el código típico de un científico de datos
# podría verse así:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# Leer dataset
data = pd.read_csv("dataset.csv")

# Preparar datos
X = data[['Age at enrollment', 'Curricular units 1st sem (enrolled)',
          'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (enrolled)',
          'Curricular units 2nd sem (approved)', 'Unemployment rate']]
y = data['Target']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear modelo de regresión logística
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
log_reg_accuracy = log_reg.score(X_test_scaled, y_test)
print(f"Logistic Regression accuracy: {log_reg_accuracy}")

# O crear modelo de Random Forest si queremos probar otra cosa
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_accuracy = rf.score(X_test_scaled, y_test)
print(f"Random Forest accuracy: {rf_accuracy}")

# O crear modelo SVM si queremos probar otra cosa
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_accuracy = svm.score(X_test_scaled, y_test)
print(f"SVM accuracy: {svm_accuracy}")

# Guardar el mejor modelo (por ejemplo, el que tenga mejor accuracy)
if log_reg_accuracy >= rf_accuracy and log_reg_accuracy >= svm_accuracy:
    best_model = log_reg
    joblib.dump(best_model, "models/best_model.pkl")
    print("Saved Logistic Regression model")
elif rf_accuracy >= log_reg_accuracy and rf_accuracy >= svm_accuracy:
    best_model = rf
    joblib.dump(best_model, "models/best_model.pkl")
    print("Saved Random Forest model")
else:
    best_model = svm
    joblib.dump(best_model, "models/best_model.pkl")
    print("Saved SVM model")

# Hacer predicciones con el mejor modelo
def predict(features):
    # Cargar el modelo
    model = joblib.load("models/best_model.pkl")
    # Escalar las características
    features_scaled = scaler.transform([features])
    # Hacer predicción
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    return {"prediction": int(prediction), "probability": float(probability)}
"""

# -------------------------------------------------------------------------
# IMPLEMENTACIÓN CON PATRÓN FACTORY
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from abc import ABC, abstractmethod


class ModelInterface(ABC):
    """Interfaz abstracta para todos los modelos de predicción"""

    @abstractmethod
    def train(self, X, y):
        """Entrena el modelo con los datos proporcionados"""
        pass

    @abstractmethod
    def predict(self, X):
        """Realiza predicciones con el modelo entrenado"""
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """Evalúa el modelo con datos de prueba"""
        pass

    @abstractmethod
    def save(self, path):
        """Guarda el modelo en la ruta especificada"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        """Carga un modelo desde la ruta especificada"""
        pass


class LogisticRegressionModel(ModelInterface):
    """Implementación del modelo de Regresión Logística"""

    def __init__(self, max_iter=1000, random_state=42):
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.scaler = StandardScaler()
        self._is_trained = False

    def train(self, X, y):
        """Entrena el modelo de regresión logística"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_trained = True
        return self

    def predict(self, X):
        """Realiza predicciones con el modelo entrenado"""
        if not self._is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")

        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "model_type": "logistic_regression"
        }

    def evaluate(self, X, y):
        """Evalúa el modelo con datos de prueba"""
        if not self._is_trained:
            raise ValueError("El modelo debe ser entrenado antes de evaluarlo")

        X_scaled = self.scaler.transform(X)
        accuracy = self.model.score(X_scaled, y)
        return {"accuracy": accuracy, "model_type": "logistic_regression"}

    def save(self, path):
        """Guarda el modelo entrenado y el scaler"""
        if not self._is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardarlo")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": "logistic_regression"
        }
        joblib.dump(model_data, path)
        return path

    @classmethod
    def load(cls, path):
        """Carga un modelo desde la ruta especificada"""
        model_data = joblib.load(path)
        if model_data["model_type"] != "logistic_regression":
            raise ValueError("El archivo no contiene un modelo de regresión logística")

        instance = cls()
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance._is_trained = True
        return instance


class RandomForestModel(ModelInterface):
    """Implementación del modelo Random Forest"""

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.scaler = StandardScaler()
        self._is_trained = False

    def train(self, X, y):
        """Entrena el modelo Random Forest"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_trained = True
        return self

    def predict(self, X):
        """Realiza predicciones con el modelo entrenado"""
        if not self._is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")

        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "model_type": "random_forest"
        }

    def evaluate(self, X, y):
        """Evalúa el modelo con datos de prueba"""
        if not self._is_trained:
            raise ValueError("El modelo debe ser entrenado antes de evaluarlo")

        X_scaled = self.scaler.transform(X)
        accuracy = self.model.score(X_scaled, y)
        return {"accuracy": accuracy, "model_type": "random_forest"}

    def save(self, path):
        """Guarda el modelo entrenado y el scaler"""
        if not self._is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardarlo")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": "random_forest"
        }
        joblib.dump(model_data, path)
        return path

    @classmethod
    def load(cls, path):
        """Carga un modelo desde la ruta especificada"""
        model_data = joblib.load(path)
        if model_data["model_type"] != "random_forest":
            raise ValueError("El archivo no contiene un modelo Random Forest")

        instance = cls()
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance._is_trained = True
        return instance


class SVMModel(ModelInterface):
    """Implementación del modelo SVM"""

    def __init__(self, probability=True, random_state=42):
        self.model = SVC(probability=probability, random_state=random_state)
        self.scaler = StandardScaler()
        self._is_trained = False

    def train(self, X, y):
        """Entrena el modelo SVM"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_trained = True
        return self

    def predict(self, X):
        """Realiza predicciones con el modelo entrenado"""
        if not self._is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")

        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "model_type": "svm"
        }

    def evaluate(self, X, y):
        """Evalúa el modelo con datos de prueba"""
        if not self._is_trained:
            raise ValueError("El modelo debe ser entrenado antes de evaluarlo")

        X_scaled = self.scaler.transform(X)
        accuracy = self.model.score(X_scaled, y)
        return {"accuracy": accuracy, "model_type": "svm"}

    def save(self, path):
        """Guarda el modelo entrenado y el scaler"""
        if not self._is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardarlo")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": "svm"
        }
        joblib.dump(model_data, path)
        return path

    @classmethod
    def load(cls, path):
        """Carga un modelo desde la ruta especificada"""
        model_data = joblib.load(path)
        if model_data["model_type"] != "svm":
            raise ValueError("El archivo no contiene un modelo SVM")

        instance = cls()
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance._is_trained = True
        return instance


class ModelFactory:
    """
    Fábrica para crear diferentes tipos de modelos de predicción.

    Esta clase implementa el patrón Factory para crear instancias
    de diferentes tipos de modelos de machine learning para la
    predicción de deserción estudiantil.
    """

    @staticmethod
    def create(model_type, **kwargs):
        """
        Crea un modelo de predicción según el tipo especificado.

        Args:
            model_type (str): Tipo de modelo a crear ('logistic', 'random_forest', 'svm')
            **kwargs: Parámetros adicionales para el modelo

        Returns:
            ModelInterface: Instancia del modelo seleccionado

        Raises:
            ValueError: Si el tipo de modelo no es soportado
        """
        if model_type == "logistic":
            return LogisticRegressionModel(**kwargs)
        elif model_type == "random_forest":
            return RandomForestModel(**kwargs)
        elif model_type == "svm":
            return SVMModel(**kwargs)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    @staticmethod
    def load_model(path):
        """
        Carga un modelo desde la ruta especificada y determina su tipo.

        Args:
            path (str): Ruta al archivo del modelo

        Returns:
            ModelInterface: Instancia del modelo cargado

        Raises:
            ValueError: Si el tipo de modelo no es reconocido
        """
        try:
            # Intentar cargar los datos del modelo
            model_data = joblib.load(path)
            model_type = model_data.get("model_type")

            if model_type == "logistic_regression":
                return LogisticRegressionModel.load(path)
            elif model_type == "random_forest":
                return RandomForestModel.load(path)
            elif model_type == "svm":
                return SVMModel.load(path)
            else:
                raise ValueError(f"Tipo de modelo no reconocido: {model_type}")
        except Exception as e:
            raise ValueError(f"Error al cargar el modelo: {str(e)}")


# Ejemplo de uso del patrón Factory
def demo_factory_pattern():
    """Demostración del uso del patrón Factory"""

    # Cargar datos
    data = pd.read_csv("../../dataset.csv")

    # Preparar datos
    X = data[['Age at enrollment', 'Curricular units 1st sem (enrolled)',
              'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (enrolled)',
              'Curricular units 2nd sem (approved)', 'Unemployment rate']]
    y = data['Target']

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Crear y evaluar diferentes modelos usando la fábrica
    models = []
    for model_type in ["logistic", "random_forest", "svm"]:
        print(f"\nCreando y entrenando modelo {model_type}...")
        model = ModelFactory.create(model_type)
        model.train(X_train, y_train)
        evaluation = model.evaluate(X_test, y_test)
        print(f"Precisión del modelo {model_type}: {evaluation['accuracy']:.4f}")
        models.append((model, evaluation['accuracy']))

        # Guardar el modelo
        model_path = f"../../models/demo_{model_type}_model.pkl"
        model.save(model_path)
        print(f"Modelo guardado en: {model_path}")

    # Encontrar el mejor modelo
    best_model, best_accuracy = max(models, key=lambda x: x[1])
    print(f"\nMejor modelo: {best_model.predict([X_test.iloc[0]])['model_type']} con precisión: {best_accuracy:.4f}")

    # Cargar un modelo guardado
    model_type = best_model.predict([X_test.iloc[0]])['model_type']
    loaded_model = ModelFactory.load_model(f"../../models/demo_{model_type}_model.pkl")
    loaded_evaluation = loaded_model.evaluate(X_test, y_test)
    print(f"Precisión del modelo cargado: {loaded_evaluation['accuracy']:.4f}")

    # Hacer una predicción de ejemplo
    sample_features = X_test.iloc[0].to_dict()
    prediction = best_model.predict([X_test.iloc[0]])
    print(f"\nPredicción para muestra de ejemplo:")
    print(f"Características: {sample_features}")
    print(f"Predicción: {'Deserción' if prediction['prediction'] == 1 else 'Graduación'}")
    print(f"Probabilidad de deserción: {prediction['probability']:.4f}")


if __name__ == "__main__":
    demo_factory_pattern()
