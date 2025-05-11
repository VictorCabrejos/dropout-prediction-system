"""
Patrón Decorator - Demostración

Este archivo muestra la implementación del patrón Decorator para añadir
funcionalidades adicionales al proceso de predicción de deserción estudiantil,
como validación, logging y caché.
"""

# -------------------------------------------------------------------------
# CÓDIGO ESPAGUETI ORIGINAL (SIN PATRÓN DECORATOR)
# -------------------------------------------------------------------------
"""
# Sin aplicar el patrón Decorator, el código típico para manejar validación,
# logging y caché podría verse así:

import pandas as pd
import numpy as np
import joblib
import logging
import time
import json
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

# Cargar modelo
model = joblib.load("models/dropout_predictor.pkl")
scaler = joblib.load("models/scaler.pkl")

# Caché para almacenar resultados de predicciones anteriores
prediction_cache = {}

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

def predict_with_logging_validation_cache(features_dict: Dict[str, Any]) -> Dict[str, Any]:
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
"""

# -------------------------------------------------------------------------
# IMPLEMENTACIÓN CON PATRÓN DECORATOR
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional, Callable
import joblib
import logging
import time
import json
from abc import ABC, abstractmethod
from functools import wraps


class PredictorInterface(ABC):
    """
    Interfaz para todos los predictores y decoradores.
    Define el contrato que deben cumplir todas las implementaciones.
    """

    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza una predicción basada en las características proporcionadas.

        Args:
            features: Diccionario con las características para la predicción

        Returns:
            Dict[str, Any]: Resultado de la predicción
        """
        pass


class DropoutPredictor(PredictorInterface):
    """
    Implementación base del predictor de deserción estudiantil.
    Esta es la clase que será decorada con funcionalidades adicionales.
    """

    def __init__(self, model_path: str, scaler_path: str):
        """
        Inicializa el predictor con un modelo y un scaler.

        Args:
            model_path: Ruta al archivo del modelo
            scaler_path: Ruta al archivo del scaler
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza una predicción basada en las características proporcionadas.

        Args:
            features: Diccionario con las características para la predicción

        Returns:
            Dict[str, Any]: Resultado de la predicción con la probabilidad y la clase
        """
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


class PredictorDecorator(PredictorInterface):
    """
    Clase base para todos los decoradores del predictor.
    """

    def __init__(self, predictor: PredictorInterface):
        """
        Inicializa el decorador con un predictor.

        Args:
            predictor: El predictor a decorar
        """
        self._predictor = predictor

    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza una predicción basada en las características proporcionadas.

        Args:
            features: Diccionario con las características para la predicción

        Returns:
            Dict[str, Any]: Resultado de la predicción
        """
        pass


class ValidationDecorator(PredictorDecorator):
    """
    Decorador que añade validación de datos antes de realizar la predicción.
    """

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida las características antes de realizar la predicción.

        Args:
            features: Diccionario con las características para la predicción

        Returns:
            Dict[str, Any]: Resultado de la predicción

        Raises:
            ValueError: Si los datos no son válidos
        """
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
    """
    Decorador que añade registro de actividad a las predicciones.
    """

    def __init__(self, predictor: PredictorInterface, logger=None):
        """
        Inicializa el decorador con un predictor y un logger.

        Args:
            predictor: El predictor a decorar
            logger: El logger a utilizar (opcional)
        """
        super().__init__(predictor)

        if logger is None:
            # Configurar un logger por defecto si no se proporciona uno
            self.logger = logging.getLogger("dropout_predictor")

            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registra información sobre la predicción antes y después de realizarla.

        Args:
            features: Diccionario con las características para la predicción

        Returns:
            Dict[str, Any]: Resultado de la predicción
        """
        self.logger.info(f"Iniciando predicción con datos: {features}")
        start_time = time.time()

        try:
            # Realizar la predicción
            result = self._predictor.predict(features)

            # Registrar información sobre el resultado
            elapsed_time = time.time() - start_time
            self.logger.info(f"Predicción completada en {elapsed_time:.4f} segundos")
            self.logger.info(f"Resultado de la predicción: {result}")

            return result
        except Exception as e:
            # Registrar información sobre el error
            elapsed_time = time.time() - start_time
            self.logger.error(f"Error durante la predicción: {str(e)}")
            self.logger.error(f"La predicción falló después de {elapsed_time:.4f} segundos")

            # Re-lanzar la excepción para que sea manejada por el código cliente
            raise


class CacheDecorator(PredictorDecorator):
    """
    Decorador que añade caché de resultados para entradas repetidas.
    """

    def __init__(self, predictor: PredictorInterface, max_size: int = 100):
        """
        Inicializa el decorador con un predictor.

        Args:
            predictor: El predictor a decorar
            max_size: Tamaño máximo de la caché
        """
        super().__init__(predictor)
        self._cache = {}
        self._max_size = max_size

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Utiliza caché para evitar predicciones repetidas.

        Args:
            features: Diccionario con las características para la predicción

        Returns:
            Dict[str, Any]: Resultado de la predicción
        """
        # Crear una clave para el caché basada en los valores de las características
        # Convertimos el diccionario a una cadena JSON para usarla como clave
        cache_key = json.dumps(features, sort_keys=True)

        # Verificar si la predicción está en caché
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Si no está en caché, realizar la predicción
        result = self._predictor.predict(features)

        # Si la caché está llena, eliminar la entrada más antigua
        if len(self._cache) >= self._max_size:
            # En un sistema real, se podría utilizar una estrategia LRU
            self._cache.pop(next(iter(self._cache)))

        # Guardar en caché
        self._cache[cache_key] = result

        return result


class PerformanceMonitorDecorator(PredictorDecorator):
    """
    Decorador que mide el rendimiento de las predicciones.
    """

    def __init__(self, predictor: PredictorInterface, threshold: float = 1.0, callback: Optional[Callable] = None):
        """
        Inicializa el decorador con un predictor.

        Args:
            predictor: El predictor a decorar
            threshold: Umbral en segundos para considerar una predicción lenta
            callback: Función a llamar cuando una predicción supera el umbral
        """
        super().__init__(predictor)
        self._threshold = threshold
        self._callback = callback
        self._total_time = 0.0
        self._call_count = 0

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mide el tiempo de ejecución de la predicción.

        Args:
            features: Diccionario con las características para la predicción

        Returns:
            Dict[str, Any]: Resultado de la predicción
        """
        start_time = time.time()

        # Realizar la predicción
        result = self._predictor.predict(features)

        # Calcular el tiempo transcurrido
        elapsed_time = time.time() - start_time

        # Actualizar estadísticas
        self._total_time += elapsed_time
        self._call_count += 1

        # Si la predicción supera el umbral, ejecutar el callback si existe
        if elapsed_time > self._threshold and self._callback:
            self._callback(elapsed_time, features)

        # Añadir información de rendimiento al resultado
        result["elapsed_time"] = elapsed_time
        result["avg_prediction_time"] = self._total_time / self._call_count if self._call_count > 0 else 0

        return result


# Ejemplo de uso del patrón Decorator
def demo_decorator_pattern():
    """Demostración del uso del patrón Decorator"""

    try:
        # Crear un predictor base (simulado para la demostración)
        # En un caso real, usaríamos rutas a archivos reales
        base_predictor = DropoutPredictor(
            model_path="../../models/dropout_predictor.pkl",
            scaler_path="../../models/scaler.pkl"
        )

        print("Demostración del patrón Decorator para añadir funcionalidades\n")

        # Características de ejemplo
        student1 = {
            "Age at enrollment": 20,
            "Curricular units 1st sem (enrolled)": 6,
            "Curricular units 1st sem (approved)": 5,
            "Curricular units 2nd sem (enrolled)": 6,
            "Curricular units 2nd sem (approved)": 5,
            "Unemployment rate": 10.8
        }

        student2 = {
            "Age at enrollment": 25,
            "Curricular units 1st sem (enrolled)": 4,
            "Curricular units 1st sem (approved)": 5,  # Error: más aprobadas que matriculadas
            "Curricular units 2nd sem (enrolled)": 6,
            "Curricular units 2nd sem (approved)": 5,
            "Unemployment rate": 12.3
        }

        # 1. Predictor base sin decoradores
        print("1. Predictor base sin decoradores:")
        try:
            result1 = base_predictor.predict(student1)
            print(f"Resultado: {result1}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")

        # 2. Predictor con validación
        print("2. Predictor con validación:")
        validated_predictor = ValidationDecorator(base_predictor)

        try:
            result2 = validated_predictor.predict(student1)
            print(f"Predicción válida: {result2}")
        except ValueError as e:
            print(f"Error de validación: {str(e)}")

        try:
            result2_invalid = validated_predictor.predict(student2)
            print(f"Predicción con datos inválidos: {result2_invalid}")
        except ValueError as e:
            print(f"Error de validación detectado correctamente: {str(e)}\n")

        # 3. Predictor con validación y logging
        print("3. Predictor con validación y logging:")
        logged_predictor = LoggingDecorator(validated_predictor)

        try:
            result3 = logged_predictor.predict(student1)
            print(f"Predicción con logging: {result3}\n")
        except ValueError as e:
            print(f"Error: {str(e)}\n")

        # 4. Predictor con validación, logging y caché
        print("4. Predictor con validación, logging y caché:")
        cached_predictor = CacheDecorator(logged_predictor)

        # Primera llamada (sin caché)
        print("Primera llamada (sin caché):")
        try:
            result4a = cached_predictor.predict(student1)
            print(f"Resultado: {result4a}")
        except ValueError as e:
            print(f"Error: {str(e)}")

        # Segunda llamada (debería usar caché)
        print("\nSegunda llamada (debería usar caché):")
        try:
            result4b = cached_predictor.predict(student1)
            print(f"Resultado: {result4b}")
            print("Nota: Observe que en el log no aparece 'Iniciando predicción' por segunda vez")
        except ValueError as e:
            print(f"Error: {str(e)}")

        # 5. Predictor con todos los decoradores y monitor de rendimiento
        def slow_prediction_callback(time, features):
            print(f"ALERTA: Predicción lenta detectada ({time:.4f}s) para {features}")

        print("\n5. Predictor con todos los decoradores y monitor de rendimiento:")
        monitored_predictor = PerformanceMonitorDecorator(
            cached_predictor,
            threshold=0.001,  # Umbral bajo para demostración
            callback=slow_prediction_callback
        )

        try:
            result5 = monitored_predictor.predict(student1)
            print(f"Resultado con información de rendimiento: {result5}\n")
        except ValueError as e:
            print(f"Error: {str(e)}\n")

        # 6. Demostración de la flexibilidad del patrón Decorator
        print("6. Demostración de la flexibilidad del patrón Decorator:")
        print("Podemos cambiar el orden de los decoradores o usar solo algunos:")

        # Primero caché, luego logging (orden diferente)
        alt_predictor1 = LoggingDecorator(CacheDecorator(base_predictor))

        # Solo monitor de rendimiento, sin validación ni caché
        alt_predictor2 = PerformanceMonitorDecorator(base_predictor)

        print("Diferentes configuraciones de decoradores creadas con éxito.\n")

        # 7. Comparación de rendimiento con y sin caché
        print("7. Comparación de rendimiento con y sin caché:")

        # Predictor sin caché
        no_cache_predictor = LoggingDecorator(ValidationDecorator(base_predictor))

        # Predictor con caché
        with_cache_predictor = CacheDecorator(LoggingDecorator(ValidationDecorator(base_predictor)))

        # Medir tiempo sin caché
        start_time = time.time()
        for _ in range(1000):
            no_cache_predictor.predict(student1)
        no_cache_time = time.time() - start_time

        # Medir tiempo con caché
        start_time = time.time()
        for _ in range(1000):
            with_cache_predictor.predict(student1)
        with_cache_time = time.time() - start_time

        print(f"Tiempo para 1000 predicciones sin caché: {no_cache_time:.4f}s")
        print(f"Tiempo para 1000 predicciones con caché: {with_cache_time:.4f}s")
        print(f"Mejora de rendimiento: {(1 - with_cache_time/no_cache_time)*100:.2f}%")

    except Exception as e:
        print(f"Error en la demostración: {str(e)}")


if __name__ == "__main__":
    demo_decorator_pattern()
