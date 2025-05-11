# Ejemplos de Implementación de los Patrones de Diseño

Este documento contiene ejemplos prácticos y concretos para implementar los patrones de diseño en el sistema de predicción de deserción estudiantil.

## Patrón Factory - Ejemplo de Implementación

El siguiente ejemplo muestra cómo implementar una fábrica de modelos que permitiría elegir entre diferentes algoritmos:

```python
# app/models/model_factory.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd

class ModelFactory:
    """
    Fábrica para crear diferentes tipos de modelos de predicción
    """

    @staticmethod
    def create(model_type: str, **kwargs):
        """
        Crea un modelo de predicción según el tipo especificado.

        Args:
            model_type (str): Tipo de modelo a crear ('logistic', 'random_forest', 'svm')
            **kwargs: Parámetros adicionales para el modelo

        Returns:
            object: Instancia del modelo seleccionado
        """
        if model_type == "logistic":
            return LogisticRegression(max_iter=1000, random_state=42, **kwargs)
        elif model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
        elif model_type == "svm":
            return SVC(probability=True, random_state=42, **kwargs)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")

# Ejemplo de uso:
# model = ModelFactory.create("random_forest", n_estimators=200)
```

## Patrón Adapter - Ejemplo de Implementación

Este ejemplo muestra cómo implementar un adaptador que permitiría manejar diferentes formatos de datos de entrada:

```python
# app/models/adapters.py
import pandas as pd
from typing import Dict, Any, Union
from ..models.schemas import PredictionInput

class StudentInputAdapter:
    """
    Adapta diferentes formatos de entrada de datos de estudiantes
    al formato esperado por los modelos de predicción.
    """

    def __init__(self, data: Union[Dict[str, Any], PredictionInput]):
        """
        Inicializa el adaptador con los datos proporcionados

        Args:
            data: Datos de entrada, pueden ser un diccionario o un objeto PredictionInput
        """
        self.data = data

    def to_model_format(self) -> Dict[str, Any]:
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
            mapping = {
                "Age at enrollment": ["age_at_enrollment", "age", "enrollment_age"],
                "Curricular units 1st sem (enrolled)": ["curricular_units_1st_sem_enrolled", "enrolled_1", "first_sem_enrolled"],
                "Curricular units 1st sem (approved)": ["curricular_units_1st_sem_approved", "approved_1", "first_sem_approved"],
                "Curricular units 2nd sem (enrolled)": ["curricular_units_2nd_sem_enrolled", "enrolled_2", "second_sem_enrolled"],
                "Curricular units 2nd sem (approved)": ["curricular_units_2nd_sem_approved", "approved_2", "second_sem_approved"],
                "Unemployment rate": ["unemployment_rate", "unemployment", "jobless_rate"]
            }

            result = {}
            for model_key, possible_keys in mapping.items():
                for key in possible_keys:
                    if key in self.data:
                        result[model_key] = self.data[key]
                        break
                if model_key not in result:
                    # Valor por defecto si no se encuentra ninguna clave
                    result[model_key] = 0

            return result
        else:
            raise ValueError("Formato de datos no soportado")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convierte los datos a un DataFrame de pandas.

        Returns:
            pd.DataFrame: DataFrame con el formato correcto para el modelo
        """
        return pd.DataFrame([self.to_model_format()])

# Ejemplo de uso:
# from ..models.schemas import PredictionInput
#
# # Desde la API (objeto Pydantic)
# api_data = PredictionInput(age_at_enrollment=20,
#                           curricular_units_1st_sem_enrolled=6,
#                           curricular_units_1st_sem_approved=5,
#                           curricular_units_2nd_sem_enrolled=6,
#                           curricular_units_2nd_sem_approved=5,
#                           unemployment_rate=10.8)
# adapter1 = StudentInputAdapter(api_data)
# df1 = adapter1.to_dataframe()
#
# # Desde otra fuente (diccionario)
# dict_data = {
#     "age": 22,
#     "enrolled_1": 7,
#     "approved_1": 6,
#     "enrolled_2": 7,
#     "approved_2": 5,
#     "unemployment": 12.3
# }
# adapter2 = StudentInputAdapter(dict_data)
# df2 = adapter2.to_dataframe()
```

## Patrón Decorator - Ejemplo de Implementación

Este ejemplo muestra cómo implementar decoradores para añadir funcionalidades como validación, logging y caché:

```python
# app/models/decorators.py
import logging
import time
from functools import wraps
from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePredictor(ABC):
    """Interfaz común para todos los predictores"""

    @abstractmethod
    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza una predicción basada en las características proporcionadas

        Args:
            features_dict: Diccionario con las características para la predicción

        Returns:
            dict: Resultado de la predicción
        """
        pass

class PredictorDecorator(BasePredictor):
    """Clase base para todos los decoradores de predictor"""

    def __init__(self, predictor: BasePredictor):
        """
        Inicializa el decorador con un predictor

        Args:
            predictor: El predictor a decorar
        """
        self._predictor = predictor

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delega la predicción al predictor decorado

        Args:
            features_dict: Diccionario con las características para la predicción

        Returns:
            dict: Resultado de la predicción
        """
        return self._predictor.predict(features_dict)

class ValidationDecorator(PredictorDecorator):
    """Añade validación avanzada antes de la predicción"""

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida los datos antes de realizar la predicción

        Args:
            features_dict: Diccionario con las características para la predicción

        Returns:
            dict: Resultado de la predicción

        Raises:
            ValueError: Si los datos no son válidos
        """
        # Validación de valores negativos
        for key, value in features_dict.items():
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"La característica '{key}' no puede ser negativa")

        # Validación de relaciones entre características
        enrolled_1 = features_dict.get("Curricular units 1st sem (enrolled)", 0)
        approved_1 = features_dict.get("Curricular units 1st sem (approved)", 0)

        enrolled_2 = features_dict.get("Curricular units 2nd sem (enrolled)", 0)
        approved_2 = features_dict.get("Curricular units 2nd sem (approved)", 0)

        if approved_1 > enrolled_1:
            raise ValueError("El número de unidades aprobadas no puede ser mayor que las matriculadas (1er semestre)")

        if approved_2 > enrolled_2:
            raise ValueError("El número de unidades aprobadas no puede ser mayor que las matriculadas (2do semestre)")

        if features_dict.get("Age at enrollment", 0) < 16:
            raise ValueError("La edad debe ser al menos 16 años")

        return self._predictor.predict(features_dict)

class LoggingDecorator(PredictorDecorator):
    """Añade registro de actividad a las predicciones"""

    def __init__(self, predictor: BasePredictor, logger=None):
        """
        Inicializa el decorador con un predictor y un logger

        Args:
            predictor: El predictor a decorar
            logger: El logger a utilizar (opcional)
        """
        super().__init__(predictor)
        self.logger = logger or logging.getLogger(__name__)

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registra información sobre la predicción

        Args:
            features_dict: Diccionario con las características para la predicción

        Returns:
            dict: Resultado de la predicción
        """
        self.logger.info(f"Iniciando predicción con datos: {features_dict}")
        start_time = time.time()

        try:
            result = self._predictor.predict(features_dict)

            elapsed_time = time.time() - start_time
            self.logger.info(f"Predicción completada en {elapsed_time:.4f} segundos")
            self.logger.info(f"Resultado de la predicción: {result}")

            return result
        except Exception as e:
            self.logger.error(f"Error durante la predicción: {str(e)}")
            raise

class CacheDecorator(PredictorDecorator):
    """Añade caché de resultados para entradas repetidas"""

    def __init__(self, predictor: BasePredictor, max_size: int = 100):
        """
        Inicializa el decorador con un predictor

        Args:
            predictor: El predictor a decorar
            max_size: Tamaño máximo de la caché
        """
        super().__init__(predictor)
        self._cache = {}
        self._max_size = max_size

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Utiliza caché para evitar predicciones repetidas

        Args:
            features_dict: Diccionario con las características para la predicción

        Returns:
            dict: Resultado de la predicción
        """
        # Crear una clave para el caché basada en los valores de las características
        # Convertimos el diccionario a una tupla de pares (clave, valor) ordenados
        cache_key = tuple(sorted(features_dict.items()))

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Si la caché está llena, eliminar la entrada más antigua
        if len(self._cache) >= self._max_size:
            # En un sistema real, se podría utilizar una estrategia LRU
            self._cache.pop(next(iter(self._cache)))

        result = self._predictor.predict(features_dict)
        self._cache[cache_key] = result
        return result

# Ejemplo de uso:
# from ..models.predictor import DropoutPredictor
#
# # Adaptar DropoutPredictor para cumplir con la interfaz BasePredictor
# class DropoutPredictorAdapter(BasePredictor):
#     def __init__(self, predictor):
#         self._predictor = predictor
#
#     def predict(self, features_dict):
#         return self._predictor.predict(features_dict)
#
# # Crear el predictor base
# base_predictor = DropoutPredictor(model_path="models/dropout_predictor.pkl")
# predictor_adapter = DropoutPredictorAdapter(base_predictor)
#
# # Aplicar decoradores
# validated_predictor = ValidationDecorator(predictor_adapter)
# logging_predictor = LoggingDecorator(validated_predictor)
# cached_predictor = CacheDecorator(logging_predictor)
#
# # Usar el predictor decorado
# result = cached_predictor.predict({
#     "Age at enrollment": 20,
#     "Curricular units 1st sem (enrolled)": 6,
#     "Curricular units 1st sem (approved)": 5,
#     "Curricular units 2nd sem (enrolled)": 6,
#     "Curricular units 2nd sem (approved)": 5,
#     "Unemployment rate": 10.8
# })
```

## Patrón Facade - Ejemplo de Implementación

Este ejemplo muestra cómo implementar una fachada que simplifica la interacción con el sistema de predicción:

```python
# app/models/facade.py
import logging
import os
from typing import Dict, Any, Union, Optional

from .adapters import StudentInputAdapter
from .decorators import ValidationDecorator, LoggingDecorator, CacheDecorator, BasePredictor
from .model_factory import ModelFactory
from .predictor import DropoutPredictor

class PredictionFacade:
    """
    Fachada que simplifica el uso del sistema de predicción
    """

    def __init__(self,
                 model_type: str = "logistic",
                 with_validation: bool = True,
                 with_logging: bool = True,
                 with_cache: bool = True,
                 model_path: Optional[str] = None,
                 dataset_path: Optional[str] = None):
        """
        Inicializa la fachada con la configuración deseada

        Args:
            model_type: Tipo de modelo a utilizar
            with_validation: Activar validación
            with_logging: Activar registro de actividad
            with_cache: Activar caché de resultados
            model_path: Ruta al modelo pre-entrenado
            dataset_path: Ruta al dataset para entrenamiento
        """
        # Configurar logging
        self.logger = logging.getLogger("prediction_system")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Crear o cargar el predictor
        self._initialize_predictor(model_type, model_path, dataset_path)

        # Aplicar decoradores según la configuración
        if with_validation:
            self.predictor = ValidationDecorator(self.predictor)

        if with_logging:
            self.predictor = LoggingDecorator(self.predictor, self.logger)

        if with_cache:
            self.predictor = CacheDecorator(self.predictor)

    def _initialize_predictor(self, model_type: str, model_path: Optional[str], dataset_path: Optional[str]):
        """
        Inicializa el predictor base

        Args:
            model_type: Tipo de modelo a utilizar
            model_path: Ruta al modelo pre-entrenado
            dataset_path: Ruta al dataset para entrenamiento
        """
        # Adaptador para DropoutPredictor que cumpla con la interfaz BasePredictor
        class DropoutPredictorAdapter(BasePredictor):
            def __init__(self, predictor):
                self._predictor = predictor

            def predict(self, features_dict):
                return self._predictor.predict(features_dict)

            def train(self, dataset_path):
                return self._predictor.train(dataset_path)

            def save_model(self, model_path):
                return self._predictor.save_model(model_path)

            @property
            def model_path(self):
                return self._predictor.model_path

        # Si hay una ruta de modelo, cargar el modelo existente
        if model_path and os.path.exists(model_path):
            self.logger.info(f"Cargando modelo desde {model_path}")
            base_predictor = DropoutPredictor(model_path=model_path)
        else:
            # Si no hay modelo, crear uno nuevo
            self.logger.info(f"Creando nuevo predictor con modelo {model_type}")
            if model_type != "logistic":
                # Si no es el modelo por defecto, crear el modelo con la factory
                model = ModelFactory.create(model_type)
                base_predictor = DropoutPredictor()
                base_predictor.model = model

                # Si hay dataset, entrenar el modelo
                if dataset_path:
                    self.logger.info(f"Entrenando modelo con dataset {dataset_path}")
                    base_predictor.train(dataset_path)

                    # Si hay ruta para guardar, guardar el modelo
                    if model_path:
                        self.logger.info(f"Guardando modelo en {model_path}")
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        base_predictor.save_model(model_path)
            else:
                # Usar el predictor por defecto
                base_predictor = DropoutPredictor()

                # Si hay dataset, entrenar el modelo
                if dataset_path:
                    self.logger.info(f"Entrenando modelo con dataset {dataset_path}")
                    base_predictor.train(dataset_path)

                    # Si hay ruta para guardar, guardar el modelo
                    if model_path:
                        self.logger.info(f"Guardando modelo en {model_path}")
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        base_predictor.save_model(model_path)

        # Adaptar el predictor para cumplir con la interfaz BasePredictor
        self.predictor = DropoutPredictorAdapter(base_predictor)

    def train(self, dataset_path: str) -> float:
        """
        Entrena el modelo con el dataset proporcionado

        Args:
            dataset_path: Ruta al archivo del dataset

        Returns:
            float: Precisión del modelo
        """
        # Acceder al predictor base a través de la cadena de decoradores
        predictor = self.predictor
        while hasattr(predictor, '_predictor'):
            predictor = predictor._predictor

        return predictor.train(dataset_path)

    def predict(self, input_data: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        Realiza una predicción a partir de los datos de entrada

        Args:
            input_data: Datos de entrada (puede ser objeto Pydantic, dict, etc.)

        Returns:
            dict: Resultado de la predicción
        """
        # Adaptar los datos de entrada
        adapter = StudentInputAdapter(input_data)
        features_dict = adapter.to_model_format()

        # Realizar la predicción
        return self.predictor.predict(features_dict)

    def save_model(self, model_path: str) -> None:
        """
        Guarda el modelo en la ruta especificada

        Args:
            model_path: Ruta donde guardar el modelo
        """
        # Acceder al predictor base a través de la cadena de decoradores
        predictor = self.predictor
        while hasattr(predictor, '_predictor'):
            predictor = predictor._predictor

        return predictor.save_model(model_path)

# Ejemplo de uso:
# from ..models.schemas import PredictionInput
#
# # Crear una fachada con la configuración deseada
# facade = PredictionFacade(
#     model_type="random_forest",
#     with_validation=True,
#     with_logging=True,
#     with_cache=True,
#     model_path="models/rf_dropout_predictor.pkl",
#     dataset_path="dataset.csv"
# )
#
# # Realizar una predicción con datos de la API
# api_data = PredictionInput(
#     age_at_enrollment=20,
#     curricular_units_1st_sem_enrolled=6,
#     curricular_units_1st_sem_approved=5,
#     curricular_units_2nd_sem_enrolled=6,
#     curricular_units_2nd_sem_approved=5,
#     unemployment_rate=10.8
# )
# result1 = facade.predict(api_data)
#
# # Realizar una predicción con datos de otra fuente
# dict_data = {
#     "age": 22,
#     "enrolled_1": 7,
#     "approved_1": 6,
#     "enrolled_2": 7,
#     "approved_2": 5,
#     "unemployment": 12.3
# }
# result2 = facade.predict(dict_data)
#
# # Guardar el modelo en un nuevo archivo
# facade.save_model("models/new_dropout_predictor.pkl")
```

## Actualización de la API para Usar la Fachada

Finalmente, este ejemplo muestra cómo actualizar el endpoint de la API para utilizar la fachada:

```python
# app/api/prediction.py
from fastapi import APIRouter, Depends, HTTPException
from ..models.schemas import PredictionInput, PredictionOutput, TrainingResult
from ..models.facade import PredictionFacade
import os

router = APIRouter(prefix="/api", tags=["prediction"])

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "models", "dropout_predictor.pkl")
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "dataset.csv")

def get_prediction_facade():
    """
    Dependency that provides the PredictionFacade instance
    """
    try:
        # Crear la fachada con la configuración deseada
        return PredictionFacade(
            model_type="logistic",  # Se puede cambiar a "random_forest" o "svm"
            with_validation=True,
            with_logging=True,
            with_cache=True,
            model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None,
            dataset_path=DATASET_PATH if not os.path.exists(MODEL_PATH) else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize prediction system: {str(e)}")

@router.post("/predict", response_model=PredictionOutput)
def predict(
    input_data: PredictionInput,
    facade: PredictionFacade = Depends(get_prediction_facade)
):
    """
    Make a prediction based on the input features
    """
    try:
        # La fachada se encarga de la adaptación de datos y la predicción
        result = facade.predict(input_data)
        return result
    except ValueError as e:
        # Errores de validación
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Otros errores
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/train", response_model=TrainingResult)
def train_model(
    facade: PredictionFacade = Depends(get_prediction_facade)
):
    """
    Train the model using the dataset
    """
    try:
        # La fachada se encarga del entrenamiento y guardado del modelo
        accuracy = facade.train(DATASET_PATH)
        facade.save_model(MODEL_PATH)

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

@router.post("/predict_with_model/{model_type}", response_model=PredictionOutput)
def predict_with_model(
    model_type: str,
    input_data: PredictionInput
):
    """
    Make a prediction using a specific model type

    Args:
        model_type: Tipo de modelo a utilizar ('logistic', 'random_forest', 'svm')
        input_data: Datos de entrada para la predicción
    """
    try:
        # Validar el tipo de modelo
        if model_type not in ["logistic", "random_forest", "svm"]:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

        # Crear una nueva fachada con el tipo de modelo especificado
        facade = PredictionFacade(
            model_type=model_type,
            with_validation=True,
            with_logging=True,
            with_cache=True,
            model_path=None,  # No cargar modelo existente
            dataset_path=DATASET_PATH  # Entrenar nuevo modelo
        )

        # Guardar el modelo con un nombre específico para el tipo
        model_type_path = os.path.join(os.path.dirname(MODEL_PATH), f"{model_type}_dropout_predictor.pkl")
        os.makedirs(os.path.dirname(model_type_path), exist_ok=True)
        facade.save_model(model_type_path)

        # Realizar la predicción
        result = facade.predict(input_data)
        return result
    except ValueError as e:
        # Errores de validación
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Otros errores
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
```
