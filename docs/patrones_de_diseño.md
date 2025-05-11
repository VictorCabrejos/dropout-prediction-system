# Aplicación de Patrones de Diseño para Escalabilidad y Flexibilidad

Este documento explica cómo se pueden aplicar los patrones de diseño Factory, Adapter, Decorator y Facade al proyecto de Predicción de Deserción Estudiantil para mejorar su escalabilidad y flexibilidad.

## Introducción

El sistema actual de predicción de deserción estudiantil está construido con un enfoque monolítico utilizando un modelo de regresión logística. Para hacerlo más escalable y flexible, podemos aplicar patrones de diseño que permitan:

- **Escalabilidad**: Capacidad para crecer en complejidad o uso (nuevos modelos, múltiples usuarios, alto volumen) sin volverse inestable o difícil de mantener.
- **Flexibilidad**: Capacidad para adaptarse a cambios (cambio de modelos, modificación de entradas, adición de logging o validación) sin reescribir la lógica central.

## Estructura Actual del Proyecto

Actualmente, el proyecto tiene estas áreas clave que pueden beneficiarse de los patrones de diseño:

1. Clase `DropoutPredictor` que maneja un único modelo de regresión logística
2. Endpoints de API que manejan directamente las entradas y salidas
3. Validaciones básicas a través de Pydantic

## Aplicación de Patrones de Diseño

### 1. Patrón Factory 🏭

#### ¿Dónde aplicarlo?
La clase `DropoutPredictor` actualmente está limitada a utilizar `LogisticRegression`. Con el patrón Factory podríamos permitir múltiples modelos de predicción como Random Forest o SVM.

#### Propuesta de implementación:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class ModelFactory:
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
            return RandomForestClassifier(random_state=42, **kwargs)
        elif model_type == "svm":
            return SVC(probability=True, random_state=42, **kwargs)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")

class PredictorFactory:
    @staticmethod
    def create(predictor_type: str, model_path=None, **kwargs):
        """
        Crea un predictor según el tipo especificado.

        Args:
            predictor_type (str): Tipo de predictor ('default', 'ensemble')
            model_path (str): Ruta al modelo pre-entrenado
            **kwargs: Parámetros adicionales

        Returns:
            object: Instancia del predictor
        """
        if predictor_type == "default":
            return DropoutPredictor(model_path=model_path, **kwargs)
        elif predictor_type == "ensemble":
            return EnsembleDropoutPredictor(model_path=model_path, **kwargs)
        else:
            raise ValueError(f"Tipo de predictor no soportado: {predictor_type}")
```

#### Beneficios:
- Facilita la experimentación con diferentes algoritmos
- Permite seleccionar el modelo óptimo para diferentes escenarios
- Centraliza la creación de objetos complejos

### 2. Patrón Adapter 🔌

#### ¿Dónde aplicarlo?
En la API actualmente se realiza una adaptación manual de los datos de entrada. Podemos encapsular esta lógica en un adaptador que convierta los diferentes formatos de entrada al formato esperado por los modelos.

#### Propuesta de implementación:

```python
import pandas as pd

class StudentInputAdapter:
    """
    Adapta diferentes formatos de entrada de datos de estudiantes
    al formato esperado por los modelos de predicción.
    """

    def __init__(self, data):
        self.data = data

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
            # Intenta diferentes formatos de nombre de campo
            age_key = next((k for k in ["age_at_enrollment", "age", "enrollment_age"]
                          if k in self.data), None)

            return {
                "Age at enrollment": self.data.get(age_key, 0),
                "Curricular units 1st sem (enrolled)": self.data.get("curricular_units_1st_sem_enrolled",
                                                                  self.data.get("enrolled_1", 0)),
                "Curricular units 1st sem (approved)": self.data.get("curricular_units_1st_sem_approved",
                                                                  self.data.get("approved_1", 0)),
                "Curricular units 2nd sem (enrolled)": self.data.get("curricular_units_2nd_sem_enrolled",
                                                                  self.data.get("enrolled_2", 0)),
                "Curricular units 2nd sem (approved)": self.data.get("curricular_units_2nd_sem_approved",
                                                                  self.data.get("approved_2", 0)),
                "Unemployment rate": self.data.get("unemployment_rate",
                                               self.data.get("unemployment", 0))
            }
        else:
            raise ValueError("Formato de datos no soportado")

    def to_dataframe(self):
        """
        Convierte los datos a un DataFrame de pandas.

        Returns:
            pd.DataFrame: DataFrame con el formato correcto para el modelo
        """
        return pd.DataFrame([self.to_model_format()])
```

#### Beneficios:
- Desacopla la estructura de datos de entrada del modelo
- Permite integrar datos de múltiples fuentes sin modificar el código principal
- Facilita la evolución independiente de la API y el modelo

### 3. Patrón Decorator 🎁

#### ¿Dónde aplicarlo?
El proceso de predicción actual carece de validación avanzada, registro de actividad y otras funcionalidades que podrían añadirse sin modificar el código base mediante decoradores.

#### Propuesta de implementación:

```python
import logging
import time
from functools import wraps

class BasePredictor:
    """Interfaz común para todos los predictores"""
    def predict(self, features_dict):
        pass

class PredictorDecorator(BasePredictor):
    """Clase base para todos los decoradores de predictor"""
    def __init__(self, predictor):
        self._predictor = predictor

    def predict(self, features_dict):
        return self._predictor.predict(features_dict)

class ValidationDecorator(PredictorDecorator):
    """Añade validación avanzada antes de la predicción"""
    def predict(self, features_dict):
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
    def __init__(self, predictor, logger=None):
        super().__init__(predictor)
        self.logger = logger or logging.getLogger(__name__)

    def predict(self, features_dict):
        self.logger.info(f"Iniciando predicción con datos: {features_dict}")
        start_time = time.time()

        result = self._predictor.predict(features_dict)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Predicción completada en {elapsed_time:.4f} segundos")
        self.logger.info(f"Resultado de la predicción: {result}")

        return result

class CacheDecorator(PredictorDecorator):
    """Añade caché de resultados para entradas repetidas"""
    def __init__(self, predictor):
        super().__init__(predictor)
        self._cache = {}

    def predict(self, features_dict):
        # Crear una clave para el caché basada en los valores de las características
        cache_key = tuple(sorted(features_dict.items()))

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._predictor.predict(features_dict)
        self._cache[cache_key] = result
        return result
```

#### Beneficios:
- Añade funcionalidades sin modificar el código base
- Permite activar/desactivar características como validación o logging
- Facilita el desarrollo y prueba de nuevas funcionalidades

### 4. Patrón Facade 🏛️

#### ¿Dónde aplicarlo?
Podemos crear una fachada para simplificar la interacción con el sistema de predicción, ocultando la complejidad de la creación de modelos, adaptación de datos y aplicación de decoradores.

#### Propuesta de implementación:

```python
import logging

class PredictionFacade:
    """
    Fachada que simplifica el uso del sistema de predicción
    """

    def __init__(self, model_type="logistic", with_validation=True,
                with_logging=True, with_cache=True, model_path=None):
        """
        Inicializa la fachada con la configuración deseada

        Args:
            model_type (str): Tipo de modelo a utilizar
            with_validation (bool): Activar validación
            with_logging (bool): Activar registro de actividad
            with_cache (bool): Activar caché de resultados
            model_path (str): Ruta al modelo pre-entrenado
        """
        # Configurar logging
        self.logger = logging.getLogger("prediction_system")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Crear el predictor base
        if model_path:
            self.predictor = DropoutPredictor(model_path=model_path)
        else:
            # Crear el modelo base
            model = ModelFactory.create(model_type)
            self.predictor = DropoutPredictor()
            self.predictor.model = model

        # Aplicar decoradores según la configuración
        if with_validation:
            self.predictor = ValidationDecorator(self.predictor)

        if with_logging:
            self.predictor = LoggingDecorator(self.predictor, self.logger)

        if with_cache:
            self.predictor = CacheDecorator(self.predictor)

    def train(self, dataset_path):
        """
        Entrena el modelo con el dataset proporcionado

        Args:
            dataset_path (str): Ruta al archivo del dataset

        Returns:
            float: Precisión del modelo
        """
        return self.predictor.train(dataset_path)

    def predict(self, input_data):
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

    def save_model(self, model_path):
        """
        Guarda el modelo en la ruta especificada

        Args:
            model_path (str): Ruta donde guardar el modelo
        """
        self.predictor.save_model(model_path)
```

#### Beneficios:
- Simplifica la interfaz del sistema de predicción
- Centraliza la configuración y creación de componentes
- Facilita el uso del sistema por parte de desarrolladores

## Implementación en el Proyecto Actual

Para implementar estos patrones en el proyecto actual, se recomienda seguir este enfoque:

1. **Refactorizar la clase `DropoutPredictor`** para que implemente la interfaz `BasePredictor`
2. **Crear los adaptadores** para manejar diferentes formatos de entrada
3. **Implementar los decoradores** para añadir funcionalidades como validación y logging
4. **Construir la fachada** para simplificar la interacción con el sistema
5. **Actualizar la API** para utilizar la fachada en lugar del predictor directamente

## Casos de Uso de Escalabilidad

Con estos patrones implementados, el sistema podrá manejar:

1. **Múltiples algoritmos de ML**: Fácilmente intercambiables a través de la Factory
2. **Diversos formatos de datos**: Adaptables mediante el patrón Adapter
3. **Funcionalidades añadidas**: Validación, logging, caché mediante Decorators
4. **API simple y consistente**: A través de la Facade

## Conclusión

La implementación de estos patrones de diseño permitirá que el sistema de predicción de deserción estudiantil sea más:

- **Escalable**: Pudiendo incorporar nuevos modelos y manejar mayores volúmenes de datos
- **Flexible**: Adaptándose a cambios en los requisitos sin modificar el código base
- **Mantenible**: Con una estructura modular que facilita las pruebas y el desarrollo

Al aplicar estos patrones, los estudiantes no solo aprenderán sobre modelos de machine learning, sino también sobre principios de ingeniería de software que son fundamentales para el desarrollo de sistemas escalables y flexibles.
