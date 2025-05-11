"""
Patrón Facade - Demostración

Este archivo muestra la implementación del patrón Facade para proporcionar
una interfaz simplificada que integra todos los componentes del sistema de
predicción de deserción estudiantil.
"""

# -------------------------------------------------------------------------
# CÓDIGO ESPAGUETI ORIGINAL (SIN PATRÓN FACADE)
# -------------------------------------------------------------------------
"""
# Sin aplicar el patrón Facade, el código para usar todos los componentes
# del sistema podría verse así:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import time
import json
import os
from typing import Dict, Any, List, Union

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dropout_predictor")

# Definir rutas de archivos
MODEL_PATH = "models/dropout_predictor.pkl"
SCALER_PATH = "models/scaler.pkl"
DATASET_PATH = "dataset.csv"

# Función para entrenar un modelo
def train_model(model_type="logistic"):
    logger.info(f"Entrenando modelo {model_type}...")

    try:
        # Leer dataset
        data = pd.read_csv(DATASET_PATH)

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

        # Crear modelo según el tipo
        if model_type == "logistic":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "svm":
            model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")

        # Entrenar modelo
        model.fit(X_train_scaled, y_train)

        # Evaluar modelo
        accuracy = model.score(X_test_scaled, y_test)
        logger.info(f"Precisión del modelo {model_type}: {accuracy:.4f}")

        # Guardar modelo y scaler
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        logger.info(f"Modelo guardado en {MODEL_PATH}")
        logger.info(f"Scaler guardado en {SCALER_PATH}")

        return accuracy

    except Exception as e:
        logger.error(f"Error al entrenar modelo: {str(e)}")
        raise

# Función para validar características
def validate_features(features):
    # Validar que todas las características necesarias estén presentes
    required_features = [
        "Age at enrollment",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (approved)",
        "Unemployment rate"
    ]

    missing_features = [feature for feature in required_features if feature not in features]
    if missing_features:
        error_msg = f"Faltan características necesarias: {missing_features}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validar valores negativos
    for feature, value in features.items():
        if isinstance(value, (int, float)) and value < 0:
            error_msg = f"La característica '{feature}' no puede ser negativa"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Validar relaciones entre características
    enrolled_1 = features.get("Curricular units 1st sem (enrolled)", 0)
    approved_1 = features.get("Curricular units 1st sem (approved)", 0)

    enrolled_2 = features.get("Curricular units 2nd sem (enrolled)", 0)
    approved_2 = features.get("Curricular units 2nd sem (approved)", 0)

    if approved_1 > enrolled_1:
        error_msg = "El número de unidades aprobadas no puede ser mayor que las matriculadas (1er semestre)"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if approved_2 > enrolled_2:
        error_msg = "El número de unidades aprobadas no puede ser mayor que las matriculadas (2do semestre)"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if features.get("Age at enrollment", 0) < 16:
        error_msg = "La edad debe ser al menos 16 años"
        logger.error(error_msg)
        raise ValueError(error_msg)

# Caché para almacenar resultados de predicciones anteriores
prediction_cache = {}

# Función para hacer predicciones
def predict(features_dict):
    # Iniciar tiempo para medir rendimiento
    start_time = time.time()

    logger.info(f"Iniciando predicción con datos: {features_dict}")

    try:
        # Validar características
        validate_features(features_dict)

        # Verificar si la predicción está en caché
        # Convertimos el diccionario a una cadena JSON para usarla como clave
        cache_key = json.dumps(features_dict, sort_keys=True)
        if cache_key in prediction_cache:
            logger.info(f"Predicción encontrada en caché")
            result = prediction_cache[cache_key]
            elapsed_time = time.time() - start_time
            logger.info(f"Predicción completada desde caché en {elapsed_time:.4f} segundos")
            return result

        # Cargar modelo y scaler
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Convertir a DataFrame para el modelo
        features_df = pd.DataFrame([features_dict])

        # Escalar características
        features_scaled = scaler.transform(features_df)

        # Hacer predicción
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "prediction_text": "Deserción" if prediction == 1 else "Graduación"
        }

        # Guardar en caché
        prediction_cache[cache_key] = result

        # Limitar tamaño de caché (máximo 100 entradas)
        if len(prediction_cache) > 100:
            # Eliminar la primera entrada (podría ser mejor usar LRU)
            first_key = next(iter(prediction_cache))
            del prediction_cache[first_key]

        elapsed_time = time.time() - start_time
        logger.info(f"Predicción completada en {elapsed_time:.4f} segundos")
        logger.info(f"Resultado de la predicción: {result}")

        return result
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error durante la predicción: {str(e)}")
        logger.error(f"La predicción falló después de {elapsed_time:.4f} segundos")
        raise

# Función para convertir diferentes formatos de entrada
def convert_input_format(input_data):
    # Si es un objeto Pydantic (desde la API)
    if hasattr(input_data, 'age_at_enrollment'):
        return {
            "Age at enrollment": input_data.age_at_enrollment,
            "Curricular units 1st sem (enrolled)": input_data.curricular_units_1st_sem_enrolled,
            "Curricular units 1st sem (approved)": input_data.curricular_units_1st_sem_approved,
            "Curricular units 2nd sem (enrolled)": input_data.curricular_units_2nd_sem_enrolled,
            "Curricular units 2nd sem (approved)": input_data.curricular_units_2nd_sem_approved,
            "Unemployment rate": input_data.unemployment_rate
        }

    # Si es un diccionario (posiblemente desde otra fuente)
    elif isinstance(input_data, dict):
        # Mapeos de claves alternativas
        mappings = {
            "Age at enrollment": ["age_at_enrollment", "age", "enrollment_age", "edad"],
            "Curricular units 1st sem (enrolled)": ["curricular_units_1st_sem_enrolled", "enrolled_1", "units_enrolled_sem1", "matriculados_sem1"],
            "Curricular units 1st sem (approved)": ["curricular_units_1st_sem_approved", "approved_1", "units_passed_sem1", "aprobados_sem1"],
            "Curricular units 2nd sem (enrolled)": ["curricular_units_2nd_sem_enrolled", "enrolled_2", "units_enrolled_sem2", "matriculados_sem2"],
            "Curricular units 2nd sem (approved)": ["curricular_units_2nd_sem_approved", "approved_2", "units_passed_sem2", "aprobados_sem2"],
            "Unemployment rate": ["unemployment_rate", "unemployment", "jobless_rate", "desempleo"]
        }

        result = {}
        for model_key, alt_keys in mappings.items():
            found = False
            for alt_key in alt_keys:
                if alt_key in input_data:
                    result[model_key] = input_data[alt_key]
                    found = True
                    break
            if not found:
                result[model_key] = 0  # Valor por defecto si no se encuentra

        return result

    else:
        raise ValueError(f"Formato de entrada no soportado: {type(input_data)}")

# Ejemplos de uso
if __name__ == "__main__":
    # Entrenar modelo
    train_model("random_forest")

    # Hacer una predicción con un formato específico
    prediction_result = predict({
        "Age at enrollment": 20,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 5,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (approved)": 5,
        "Unemployment rate": 10.8
    })
    print(f"Resultado: {prediction_result}")

    # Con otro formato de entrada
    alternative_format = {
        "age": 22,
        "enrolled_1": 7,
        "approved_1": 6,
        "enrolled_2": 7,
        "approved_2": 5,
        "unemployment": 12.3
    }
    converted_format = convert_input_format(alternative_format)
    prediction_result2 = predict(converted_format)
    print(f"Resultado con formato alternativo: {prediction_result2}")
"""

# -------------------------------------------------------------------------
# IMPLEMENTACIÓN CON PATRÓN FACADE
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional, Type
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import time
import json
import os
from abc import ABC, abstractmethod
from functools import wraps


# ----------------- Componentes del sistema -----------------

class ModelFactory:
    """
    Fábrica para crear diferentes tipos de modelos de predicción.
    """

    @staticmethod
    def create(model_type: str, **kwargs):
        """
        Crea un modelo de predicción según el tipo especificado.

        Args:
            model_type: Tipo de modelo a crear ('logistic', 'random_forest', 'svm')
            **kwargs: Parámetros adicionales para el modelo

        Returns:
            object: Instancia del modelo seleccionado

        Raises:
            ValueError: Si el tipo de modelo no es soportado
        """
        if model_type == "logistic":
            return LogisticRegression(max_iter=1000, random_state=42, **kwargs)
        elif model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
        elif model_type == "svm":
            return SVC(probability=True, random_state=42, **kwargs)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")


class InputAdapter:
    """
    Adaptador para convertir diferentes formatos de entrada al formato esperado por el modelo.
    """

    def __init__(self, data):
        """
        Inicializa el adaptador con los datos proporcionados.

        Args:
            data: Datos de entrada (puede ser un objeto Pydantic, un diccionario, etc.)
        """
        self.data = data

        # Mapeo de claves alternativas a las claves esperadas por el modelo
        self.key_mappings = {
            "Age at enrollment": ["age_at_enrollment", "age", "enrollment_age", "edad"],
            "Curricular units 1st sem (enrolled)": ["curricular_units_1st_sem_enrolled", "enrolled_1", "units_enrolled_sem1", "matriculados_sem1"],
            "Curricular units 1st sem (approved)": ["curricular_units_1st_sem_approved", "approved_1", "units_passed_sem1", "aprobados_sem1"],
            "Curricular units 2nd sem (enrolled)": ["curricular_units_2nd_sem_enrolled", "enrolled_2", "units_enrolled_sem2", "matriculados_sem2"],
            "Curricular units 2nd sem (approved)": ["curricular_units_2nd_sem_approved", "approved_2", "units_passed_sem2", "aprobados_sem2"],
            "Unemployment rate": ["unemployment_rate", "unemployment", "jobless_rate", "desempleo"]
        }

    def to_model_format(self):
        """
        Convierte los datos al formato esperado por el modelo.

        Returns:
            dict: Diccionario con el formato correcto de características
        """
        # Si es un objeto Pydantic (desde la API)
        if hasattr(self.data, 'age_at_enrollment'):
            return {
                "Age at enrollment": self.data.age_at_enrollment,
                "Curricular units 1st sem (enrolled)": self.data.curricular_units_1st_sem_enrolled,
                "Curricular units 1st sem (approved)": self.data.curricular_units_1st_sem_approved,
                "Curricular units 2nd sem (enrolled)": self.data.curricular_units_2nd_sem_enrolled,
                "Curricular units 2nd sem (approved)": self.data.curricular_units_2nd_sem_approved,
                "Unemployment rate": self.data.unemployment_rate
            }

        # Si es un diccionario (posiblemente desde otra fuente)
        elif isinstance(self.data, dict):
            model_features = {}

            # Para cada característica esperada por el modelo
            for model_key, alt_keys in self.key_mappings.items():
                # Buscar la primera clave alternativa que exista en el diccionario
                found = False
                for alt_key in alt_keys:
                    if alt_key in self.data:
                        model_features[model_key] = self.data[alt_key]
                        found = True
                        break

                # Si no se encontró ninguna clave alternativa para esta característica
                if not found:
                    model_features[model_key] = 0  # Valor por defecto

            return model_features
        else:
            raise ValueError("Formato de datos no soportado")

    def to_dataframe(self):
        """
        Convierte los datos a un DataFrame de pandas.

        Returns:
            pd.DataFrame: DataFrame con el formato correcto para el modelo
        """
        return pd.DataFrame([self.to_model_format()])


class BasePredictor(ABC):
    """Interfaz para todos los predictores y decoradores."""

    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza una predicción basada en las características proporcionadas."""
        pass


class DropoutPredictor(BasePredictor):
    """Implementación base del predictor de deserción estudiantil."""

    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza una predicción basada en las características proporcionadas."""
        if self.model is None or self.scaler is None:
            raise ValueError("El modelo y el scaler deben ser inicializados antes de hacer predicciones")

        # Convertir a DataFrame para el modelo
        features_df = pd.DataFrame([features])

        # Escalar características
        features_scaled = self.scaler.transform(features_df)

        # Hacer predicción
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "prediction_text": "Deserción" if prediction == 1 else "Graduación"
        }

    def train(self, X_train, y_train):
        """Entrena el modelo con los datos proporcionados."""
        if self.model is None:
            raise ValueError("El modelo debe ser inicializado antes de entrenar")

        if self.scaler is None:
            self.scaler = StandardScaler()

        # Escalar características de entrenamiento
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Entrenar modelo
        self.model.fit(X_train_scaled, y_train)

    def evaluate(self, X_test, y_test):
        """Evalúa el modelo con datos de prueba."""
        if self.model is None or self.scaler is None:
            raise ValueError("El modelo y el scaler deben ser inicializados antes de evaluar")

        # Escalar características de prueba
        X_test_scaled = self.scaler.transform(X_test)

        # Calcular precisión
        accuracy = self.model.score(X_test_scaled, y_test)

        return accuracy


class PredictorDecorator(BasePredictor):
    """Clase base para todos los decoradores del predictor."""

    def __init__(self, predictor: BasePredictor):
        self._predictor = predictor

    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        pass


class ValidationDecorator(PredictorDecorator):
    """Decorador que añade validación de datos antes de realizar la predicción."""

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        # Validar que todas las características necesarias estén presentes
        required_features = [
            "Age at enrollment",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (approved)",
            "Unemployment rate"
        ]

        missing_features = [feature for feature in required_features if feature not in features]
        if missing_features:
            raise ValueError(f"Faltan características necesarias: {', '.join(missing_features)}")

        # Validar valores negativos
        for feature, value in features.items():
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"La característica '{feature}' no puede ser negativa")

        # Validar relaciones entre características
        enrolled_1 = features.get("Curricular units 1st sem (enrolled)", 0)
        approved_1 = features.get("Curricular units 1st sem (approved)", 0)

        enrolled_2 = features.get("Curricular units 2nd sem (enrolled)", 0)
        approved_2 = features.get("Curricular units 2nd sem (approved)", 0)

        if approved_1 > enrolled_1:
            raise ValueError("El número de unidades aprobadas no puede ser mayor que las matriculadas (1er semestre)")

        if approved_2 > enrolled_2:
            raise ValueError("El número de unidades aprobadas no puede ser mayor que las matriculadas (2do semestre)")

        if features.get("Age at enrollment", 0) < 16:
            raise ValueError("La edad debe ser al menos 16 años")

        # Si la validación es exitosa, pasar la predicción al decorador interno
        return self._predictor.predict(features)


class LoggingDecorator(PredictorDecorator):
    """Decorador que añade registro de actividad a las predicciones."""

    def __init__(self, predictor: BasePredictor, logger=None):
        super().__init__(predictor)
        self.logger = logger or logging.getLogger(__name__)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Iniciando predicción con datos: {features}")
        start_time = time.time()

        try:
            result = self._predictor.predict(features)

            elapsed_time = time.time() - start_time
            self.logger.info(f"Predicción completada en {elapsed_time:.4f} segundos")
            self.logger.info(f"Resultado de la predicción: {result}")

            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Error durante la predicción: {str(e)}")
            self.logger.error(f"La predicción falló después de {elapsed_time:.4f} segundos")
            raise


class CacheDecorator(PredictorDecorator):
    """Decorador que añade caché de resultados para entradas repetidas."""

    def __init__(self, predictor: BasePredictor, max_size: int = 100):
        super().__init__(predictor)
        self._cache = {}
        self._max_size = max_size

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        # Crear una clave para el caché basada en los valores de las características
        cache_key = json.dumps(features, sort_keys=True)

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._predictor.predict(features)

        # Si la caché está llena, eliminar la entrada más antigua
        if len(self._cache) >= self._max_size:
            self._cache.pop(next(iter(self._cache)))

        self._cache[cache_key] = result
        return result


# ----------------- Implementación de la Fachada -----------------

class PredictionSystemFacade:
    """
    Fachada que simplifica el uso del sistema de predicción de deserción estudiantil.

    Esta clase proporciona una interfaz unificada para las diferentes partes del sistema,
    ocultando la complejidad de la creación de modelos, la validación de datos, el
    registro de actividad, etc.
    """

    def __init__(self,
                 model_type: str = "logistic",
                 model_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 dataset_path: Optional[str] = None,
                 with_validation: bool = True,
                 with_logging: bool = True,
                 with_cache: bool = True):
        """
        Inicializa la fachada con la configuración deseada.

        Args:
            model_type: Tipo de modelo a utilizar ('logistic', 'random_forest', 'svm')
            model_path: Ruta al archivo del modelo
            scaler_path: Ruta al archivo del scaler
            dataset_path: Ruta al archivo del dataset
            with_validation: Activar validación
            with_logging: Activar registro de actividad
            with_cache: Activar caché de resultados
        """
        # Configurar logger
        self.logger = logging.getLogger("prediction_system")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.model_type = model_type
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.dataset_path = dataset_path

        # Inicializar el predictor base
        self._init_predictor(model_path, scaler_path, model_type)

        # Aplicar decoradores según la configuración
        if with_validation:
            self.predictor = ValidationDecorator(self.predictor)

        if with_logging:
            self.predictor = LoggingDecorator(self.predictor, self.logger)

        if with_cache:
            self.predictor = CacheDecorator(self.predictor)

        # Si hay un dataset y no hay modelo cargado, entrenar un modelo nuevo
        if dataset_path and not (model_path and os.path.exists(model_path)):
            self.train_model()

    def _init_predictor(self, model_path, scaler_path, model_type):
        """
        Inicializa el predictor base con un modelo y un scaler.

        Args:
            model_path: Ruta al archivo del modelo
            scaler_path: Ruta al archivo del scaler
            model_type: Tipo de modelo a crear si no hay modelo cargado
        """
        self.predictor = DropoutPredictor()

        # Si hay rutas de modelo y scaler, intentar cargarlos
        if model_path and os.path.exists(model_path):
            try:
                self.logger.info(f"Cargando modelo desde {model_path}")
                self.predictor.model = joblib.load(model_path)
            except Exception as e:
                self.logger.error(f"Error al cargar modelo: {str(e)}")
                # Crear un modelo nuevo
                self.predictor.model = ModelFactory.create(model_type)
        else:
            # Crear un modelo nuevo
            self.predictor.model = ModelFactory.create(model_type)

        if scaler_path and os.path.exists(scaler_path):
            try:
                self.logger.info(f"Cargando scaler desde {scaler_path}")
                self.predictor.scaler = joblib.load(scaler_path)
            except Exception as e:
                self.logger.error(f"Error al cargar scaler: {str(e)}")
                # Crear un scaler nuevo
                self.predictor.scaler = StandardScaler()
        else:
            # Crear un scaler nuevo
            self.predictor.scaler = StandardScaler()

    def train_model(self, dataset_path: Optional[str] = None) -> float:
        """
        Entrena el modelo con el dataset proporcionado.

        Args:
            dataset_path: Ruta al archivo del dataset (opcional, usa el de la inicialización si no se proporciona)

        Returns:
            float: Precisión del modelo
        """
        path = dataset_path or self.dataset_path
        if not path:
            raise ValueError("Se debe proporcionar una ruta al dataset")

        self.logger.info(f"Entrenando modelo {self.model_type} con dataset {path}")

        try:
            # Leer dataset
            data = pd.read_csv(path)

            # Preparar datos
            X = data[['Age at enrollment', 'Curricular units 1st sem (enrolled)',
                    'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (enrolled)',
                    'Curricular units 2nd sem (approved)', 'Unemployment rate']]
            y = data['Target']

            # Dividir en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Acceder al predictor base a través de la cadena de decoradores
            base_predictor = self._get_base_predictor()

            # Entrenar modelo
            base_predictor.train(X_train, y_train)

            # Evaluar modelo
            accuracy = base_predictor.evaluate(X_test, y_test)
            self.logger.info(f"Precisión del modelo {self.model_type}: {accuracy:.4f}")

            return accuracy
        except Exception as e:
            self.logger.error(f"Error al entrenar modelo: {str(e)}")
            raise

    def _get_base_predictor(self) -> DropoutPredictor:
        """
        Obtiene el predictor base a través de la cadena de decoradores.

        Returns:
            DropoutPredictor: El predictor base
        """
        predictor = self.predictor
        while isinstance(predictor, PredictorDecorator):
            predictor = predictor._predictor
        return predictor

    def predict(self, input_data) -> Dict[str, Any]:
        """
        Realiza una predicción a partir de los datos de entrada.

        Args:
            input_data: Datos de entrada (puede ser un objeto Pydantic, un diccionario, etc.)

        Returns:
            Dict[str, Any]: Resultado de la predicción
        """
        # Adaptar los datos de entrada
        adapter = InputAdapter(input_data)
        features_dict = adapter.to_model_format()

        # Hacer predicción
        return self.predictor.predict(features_dict)

    def save_model(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None) -> None:
        """
        Guarda el modelo y el scaler en las rutas especificadas.

        Args:
            model_path: Ruta donde guardar el modelo
            scaler_path: Ruta donde guardar el scaler
        """
        base_predictor = self._get_base_predictor()

        # Guardar modelo
        if model_path:
            try:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump(base_predictor.model, model_path)
                self.logger.info(f"Modelo guardado en {model_path}")
            except Exception as e:
                self.logger.error(f"Error al guardar modelo: {str(e)}")

        # Guardar scaler
        if scaler_path:
            try:
                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                joblib.dump(base_predictor.scaler, scaler_path)
                self.logger.info(f"Scaler guardado en {scaler_path}")
            except Exception as e:
                self.logger.error(f"Error al guardar scaler: {str(e)}")

    def switch_model(self, model_type: str) -> None:
        """
        Cambia el tipo de modelo utilizado.

        Args:
            model_type: Tipo de modelo a utilizar ('logistic', 'random_forest', 'svm')
        """
        self.logger.info(f"Cambiando a modelo {model_type}")

        # Crear nuevo modelo
        model = ModelFactory.create(model_type)

        # Actualizar el predictor base
        base_predictor = self._get_base_predictor()
        base_predictor.model = model

        # Actualizar tipo de modelo
        self.model_type = model_type

        # Si hay un dataset, entrenar el nuevo modelo
        if self.dataset_path:
            self.train_model()

    def batch_predict(self, input_data_list: List) -> List[Dict[str, Any]]:
        """
        Realiza predicciones para una lista de datos de entrada.

        Args:
            input_data_list: Lista de datos de entrada

        Returns:
            List[Dict[str, Any]]: Lista de resultados de predicción
        """
        results = []
        for input_data in input_data_list:
            try:
                result = self.predict(input_data)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error en predicción por lotes: {str(e)}")
                # Añadir un resultado de error
                results.append({
                    "error": str(e),
                    "input": input_data
                })
        return results

    def clear_cache(self) -> None:
        """Limpia la caché de predicciones."""
        # Buscar el decorador de caché
        predictor = self.predictor
        while isinstance(predictor, PredictorDecorator):
            if isinstance(predictor, CacheDecorator):
                predictor._cache = {}
                self.logger.info("Caché limpiada")
                return
            predictor = predictor._predictor

        self.logger.warning("No se encontró caché para limpiar")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Devuelve información sobre el modelo actual.

        Returns:
            Dict[str, Any]: Información del modelo
        """
        base_predictor = self._get_base_predictor()

        # Obtener información del modelo
        model_info = {
            "model_type": self.model_type,
            "has_model": base_predictor.model is not None,
            "has_scaler": base_predictor.scaler is not None
        }

        # Añadir información adicional si está disponible
        if hasattr(base_predictor.model, 'n_features_in_'):
            model_info["n_features"] = base_predictor.model.n_features_in_

        if hasattr(base_predictor.model, 'classes_'):
            model_info["classes"] = base_predictor.model.classes_.tolist()

        if self.model_type == "random_forest" and hasattr(base_predictor.model, 'n_estimators'):
            model_info["n_estimators"] = base_predictor.model.n_estimators

        if self.model_type == "logistic" and hasattr(base_predictor.model, 'coef_'):
            model_info["has_coefficients"] = True

        return model_info


# Ejemplo de uso del patrón Facade
def demo_facade_pattern():
    """Demostración del uso del patrón Facade"""

    print("Demostración del patrón Facade para el sistema de predicción\n")

    # Rutas de archivos
    model_path = "../../models/demo_facade_model.pkl"
    scaler_path = "../../models/demo_facade_scaler.pkl"
    dataset_path = "../../dataset.csv"

    # 1. Crear una fachada con la configuración deseada
    print("1. Creación de la fachada y entrenamiento del modelo:")
    facade = PredictionSystemFacade(
        model_type="random_forest",
        dataset_path=dataset_path,
        with_validation=True,
        with_logging=True,
        with_cache=True
    )

    # La fachada se entrena automáticamente si se proporciona un dataset
    # y no hay un modelo cargado

    # Guardar el modelo y el scaler
    facade.save_model(model_path, scaler_path)

    # 2. Hacer predicciones con diferentes formatos de entrada
    print("\n2. Predicción con diferentes formatos de entrada:")

    # Diccionario con formato estándar
    student1 = {
        "Age at enrollment": 20,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (approved)": 5,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (approved)": 5,
        "Unemployment rate": 10.8
    }

    # Diccionario con formato alternativo
    student2 = {
        "age": 22,
        "enrolled_1": 7,
        "approved_1": 6,
        "enrolled_2": 7,
        "approved_2": 5,
        "unemployment": 12.3
    }

    # Diccionario con formato español
    student3 = {
        "edad": 25,
        "matriculados_sem1": 5,
        "aprobados_sem1": 4,
        "matriculados_sem2": 6,
        "aprobados_sem2": 3,
        "desempleo": 15.2
    }

    # Realizar predicciones
    result1 = facade.predict(student1)
    result2 = facade.predict(student2)
    result3 = facade.predict(student3)

    print(f"Predicción para estudiante 1: {result1}")
    print(f"Predicción para estudiante 2: {result2}")
    print(f"Predicción para estudiante 3: {result3}")

    # 3. Cambiar el modelo y reentrenar
    print("\n3. Cambio de modelo a SVM y reentrenamiento:")
    facade.switch_model("svm")

    # 4. Hacer predicciones por lotes
    print("\n4. Predicción por lotes:")
    batch_results = facade.batch_predict([student1, student2, student3])

    for i, result in enumerate(batch_results):
        print(f"Predicción para estudiante {i+1}: {result}")

    # 5. Obtener información del modelo
    print("\n5. Información del modelo actual:")
    model_info = facade.get_model_info()
    print(f"Información del modelo: {model_info}")

    # 6. Demostrar la caché
    print("\n6. Demostración de la caché:")

    # Primera llamada
    start_time = time.time()
    facade.predict(student1)
    first_call_time = time.time() - start_time

    # Segunda llamada (debería usar caché)
    start_time = time.time()
    facade.predict(student1)
    second_call_time = time.time() - start_time

    print(f"Tiempo para primera llamada: {first_call_time:.6f}s")
    print(f"Tiempo para segunda llamada (desde caché): {second_call_time:.6f}s")
    print(f"Mejora de rendimiento: {(1 - second_call_time/first_call_time)*100:.2f}%")

    # Limpiar caché
    facade.clear_cache()

    # 7. Demostrar la validación
    print("\n7. Demostración de la validación:")

    # Datos inválidos (más unidades aprobadas que matriculadas)
    invalid_student = {
        "age": 19,
        "enrolled_1": 4,
        "approved_1": 5,  # Error: más aprobadas que matriculadas
        "enrolled_2": 5,
        "approved_2": 4,
        "unemployment": 9.5
    }

    try:
        facade.predict(invalid_student)
        print("La predicción se realizó (no debería llegar aquí)")
    except ValueError as e:
        print(f"Error de validación detectado correctamente: {str(e)}")

    print("\nDemostración del patrón Facade completada")


if __name__ == "__main__":
    demo_facade_pattern()
