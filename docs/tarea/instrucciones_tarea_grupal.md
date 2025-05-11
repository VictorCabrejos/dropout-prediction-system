# Tarea Grupal: Refactorización de Código con Patrones de Diseño

## Descripción

En el mundo real, los proyectos de Machine Learning suelen empezar como prototipos rápidos o pruebas de concepto que priorizan la funcionalidad sobre la estructura. Sin embargo, cuando estos proyectos evolucionan y se convierten en sistemas en producción, enfrentan desafíos como:

- Necesidad de incorporar nuevos algoritmos de ML
- Adaptación a diferentes formatos de datos
- Agregar funcionalidades como validación, logging, monitoreo
- Mantenimiento por diferentes equipos de desarrollo

En esta tarea, ustedes actuarán como ML Engineers que han heredado un código "espagueti" de un científico de datos que desarrolló un prototipo funcional pero poco escalable para el sistema de predicción de deserción estudiantil.

## Objetivo

Su equipo debe refactorizar el código proporcionado utilizando al menos dos de los siguientes patrones de diseño:
- Factory Pattern
- Adapter Pattern
- Decorator Pattern
- Facade Pattern

## Código a Refactorizar

El siguiente código funciona correctamente, pero tiene varios problemas de diseño que lo hacen difícil de mantener, extender y escalar:

```python
# dropout_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import json
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("dropout_predictor")

# Caché para almacenar resultados previos
prediction_cache = {}

class DropoutPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = "models/dropout_predictor.pkl"
        self.features = [
            'Age at enrollment',
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (approved)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (approved)',
            'Unemployment rate'
        ]

        # Cargar modelo si existe
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)

    def train(self, dataset_path, algorithm="logistic"):
        logger.info(f"Entrenando modelo con algoritmo {algorithm}...")

        try:
            # Leer datos
            data = pd.read_csv(dataset_path)

            # Preparar datos
            X = data[self.features]
            y = data['Target']

            # Dividir en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Escalar características
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Seleccionar y entrenar modelo según el algoritmo
            if algorithm == "logistic":
                self.model = LogisticRegression(max_iter=1000, random_state=42)
            elif algorithm == "random_forest":
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                logger.error(f"Algoritmo no soportado: {algorithm}")
                raise ValueError(f"Algoritmo no soportado: {algorithm}")

            # Entrenar modelo
            self.model.fit(X_train_scaled, y_train)

            # Evaluar modelo
            accuracy = self.model.score(X_test_scaled, y_test)
            logger.info(f"Precisión del modelo: {accuracy:.4f}")

            # Guardar modelo
            self.save_model(self.model_path)

            return accuracy

        except Exception as e:
            logger.error(f"Error al entrenar modelo: {str(e)}")
            raise

    def predict(self, input_data):
        start_time = time.time()

        try:
            # Si es un objeto API (Pydantic)
            if hasattr(input_data, 'age_at_enrollment'):
                features_dict = {
                    "Age at enrollment": input_data.age_at_enrollment,
                    "Curricular units 1st sem (enrolled)": input_data.curricular_units_1st_sem_enrolled,
                    "Curricular units 1st sem (approved)": input_data.curricular_units_1st_sem_approved,
                    "Curricular units 2nd sem (enrolled)": input_data.curricular_units_2nd_sem_enrolled,
                    "Curricular units 2nd sem (approved)": input_data.curricular_units_2nd_sem_approved,
                    "Unemployment rate": input_data.unemployment_rate
                }
            # Si es un diccionario con formato diferente
            elif isinstance(input_data, dict):
                # Intentar diferentes nombres de campos
                features_dict = {}

                # Edad
                if "age_at_enrollment" in input_data:
                    features_dict["Age at enrollment"] = input_data["age_at_enrollment"]
                elif "age" in input_data:
                    features_dict["Age at enrollment"] = input_data["age"]
                elif "enrollment_age" in input_data:
                    features_dict["Age at enrollment"] = input_data["enrollment_age"]
                else:
                    raise ValueError("No se encontró información sobre la edad")

                # Unidades 1er semestre (matriculadas)
                if "curricular_units_1st_sem_enrolled" in input_data:
                    features_dict["Curricular units 1st sem (enrolled)"] = input_data["curricular_units_1st_sem_enrolled"]
                elif "enrolled_1" in input_data:
                    features_dict["Curricular units 1st sem (enrolled)"] = input_data["enrolled_1"]
                else:
                    raise ValueError("No se encontró información sobre unidades matriculadas en 1er semestre")

                # Unidades 1er semestre (aprobadas)
                if "curricular_units_1st_sem_approved" in input_data:
                    features_dict["Curricular units 1st sem (approved)"] = input_data["curricular_units_1st_sem_approved"]
                elif "approved_1" in input_data:
                    features_dict["Curricular units 1st sem (approved)"] = input_data["approved_1"]
                else:
                    raise ValueError("No se encontró información sobre unidades aprobadas en 1er semestre")

                # Unidades 2do semestre (matriculadas)
                if "curricular_units_2nd_sem_enrolled" in input_data:
                    features_dict["Curricular units 2nd sem (enrolled)"] = input_data["curricular_units_2nd_sem_enrolled"]
                elif "enrolled_2" in input_data:
                    features_dict["Curricular units 2nd sem (enrolled)"] = input_data["enrolled_2"]
                else:
                    raise ValueError("No se encontró información sobre unidades matriculadas en 2do semestre")

                # Unidades 2do semestre (aprobadas)
                if "curricular_units_2nd_sem_approved" in input_data:
                    features_dict["Curricular units 2nd sem (approved)"] = input_data["curricular_units_2nd_sem_approved"]
                elif "approved_2" in input_data:
                    features_dict["Curricular units 2nd sem (approved)"] = input_data["approved_2"]
                else:
                    raise ValueError("No se encontró información sobre unidades aprobadas en 2do semestre")

                # Tasa de desempleo
                if "unemployment_rate" in input_data:
                    features_dict["Unemployment rate"] = input_data["unemployment_rate"]
                elif "unemployment" in input_data:
                    features_dict["Unemployment rate"] = input_data["unemployment"]
                else:
                    raise ValueError("No se encontró información sobre la tasa de desempleo")
            else:
                raise ValueError(f"Formato de entrada no soportado: {type(input_data)}")

            # Validar valores
            self._validate_input(features_dict)

            # Revisar caché
            cache_key = json.dumps(features_dict, sort_keys=True)
            if cache_key in prediction_cache:
                logger.info(f"Predicción encontrada en caché")
                result = prediction_cache[cache_key]
                elapsed_time = time.time() - start_time
                logger.info(f"Predicción completada desde caché en {elapsed_time:.4f} segundos")
                return result

            # Convertir a DataFrame para el modelo
            features_df = pd.DataFrame([features_dict])

            # Verificar si tenemos todas las características necesarias
            missing_features = [f for f in self.features if f not in features_df.columns]
            if missing_features:
                logger.error(f"Faltan características: {missing_features}")
                raise ValueError(f"Faltan características: {missing_features}")

            # Verificar si el modelo está cargado
            if self.model is None:
                logger.error("El modelo no está cargado")
                raise ValueError("El modelo no está cargado")

            # Escalar características
            features_scaled = self.scaler.transform(features_df)

            # Hacer predicción
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]

            result = {
                "prediction": int(prediction),
                "probability": float(probability),
                "prediction_text": "Deserción" if prediction == 1 else "Graduación"
            }

            # Guardar en caché
            prediction_cache[cache_key] = result

            # Limitar tamaño de caché
            if len(prediction_cache) > 100:
                # Eliminar la primera entrada
                first_key = next(iter(prediction_cache))
                del prediction_cache[first_key]

            elapsed_time = time.time() - start_time
            logger.info(f"Predicción completada en {elapsed_time:.4f} segundos")
            logger.info(f"Resultado: {result}")

            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error durante la predicción: {str(e)}")
            logger.error(f"La predicción falló después de {elapsed_time:.4f} segundos")
            raise

    def _validate_input(self, features):
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

    def save_model(self, path):
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Guardar modelo y scaler juntos
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "features": self.features
        }

        joblib.dump(model_data, path)
        logger.info(f"Modelo guardado en {path}")

    def load_model(self, path):
        try:
            model_data = joblib.load(path)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.features = model_data["features"]

            logger.info(f"Modelo cargado desde {path}")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            raise

# Función para usar en aplicación FastAPI
def get_predictor():
    return DropoutPredictor()
```

## Problemas del Código Original

El código anterior tiene varios problemas:

1. **Acoplamiento alto**: La clase `DropoutPredictor` hace demasiadas cosas (maneja datos, entrena modelos, valida, cachea, etc.).
2. **Difícil extensibilidad**: Para añadir un nuevo algoritmo o formato de entrada, hay que modificar directamente la clase principal.
3. **Validación y caché mezclados**: Las funcionalidades de validación y caché están atadas al predictor.
4. **Difícil testeo**: Al tener todas las funcionalidades juntas, es complicado probar cada una por separado.

## Instrucciones

1. Formen grupos (no hay límite específico de miembros)
2. Seleccionen al menos 2 de los 4 patrones de diseño mencionados
3. Refactoricen el código aplicando estos patrones
4. Documenten claramente:
   - Por qué seleccionaron esos patrones
   - Cómo mejoran la escalabilidad y flexibilidad del código
   - Qué ventajas ofrecen para un entorno de producción real

## Pistas para la Refactorización

### Factory Pattern
- Consideren crear una clase `ModelFactory` que pueda encapsular la creación de diferentes algoritmos de ML
- Posible ubicación: `app/models/factory.py`
- Clases potenciales: `ModelInterface`, `LogisticRegressionModel`, `RandomForestModel`

### Adapter Pattern
- Piensen en una clase `InputAdapter` que pueda normalizar diferentes formatos de entrada
- Posible ubicación: `app/models/adapters.py`
- Clases potenciales: `InputAdapter`, `ApiInputAdapter`, `DictionaryInputAdapter`

### Decorator Pattern
- Consideren separar funcionalidades como validación y caché en decoradores
- Posible ubicación: `app/models/decorators.py`
- Clases potenciales: `PredictorDecorator`, `ValidationDecorator`, `CacheDecorator`, `LoggingDecorator`

### Facade Pattern
- Piensen en una interfaz simplificada que oculte la complejidad del sistema
- Posible ubicación: `app/models/facade.py`
- Clases potenciales: `PredictionSystemFacade`

## Entregables

Para el próximo domingo, deben entregar:

1. Código refactorizado
2. Un documento explicando:
   - Patrones seleccionados y justificación
   - Diagrama de clases de la nueva estructura
   - Explicación de cómo los cambios mejoran la escalabilidad y flexibilidad
   - Escenarios reales donde esta refactorización sería valiosa

## Criterios de Evaluación

- Correcta aplicación de los patrones de diseño (40%)
- Calidad y claridad del código refactorizado (30%)
- Justificación de las decisiones de diseño (20%)
- Documentación (10%)

## Consejos desde la Perspectiva de un ML Engineer

Como ML Engineers, es crucial entender que el código de producción para sistemas de ML debe ser:

- **Adaptable**: Los algoritmos y técnicas de ML evolucionan rápidamente
- **Mantenible**: Diferentes equipos necesitarán entender y modificar el código
- **Testeable**: Cada componente debe poder probarse de forma aislada
- **Extensible**: Debe ser fácil añadir nuevos modelos, fuentes de datos o funcionalidades
- **Robusto**: Incluir validaciones, logging, y manejo de errores apropiados

Estos patrones de diseño son herramientas esenciales para transformar prototipos de ML en sistemas robustos de producción.

¡Buena suerte en la refactorización!
