"""
Patrón Adapter - Demostración

Este archivo muestra la implementación del patrón Adapter para normalizar
diferentes formatos de entrada al formato esperado por el modelo de predicción
de deserción estudiantil.
"""

# -------------------------------------------------------------------------
# CÓDIGO ESPAGUETI ORIGINAL (SIN PATRÓN ADAPTER)
# -------------------------------------------------------------------------
"""
# Sin aplicar el patrón Adapter, el código típico de un científico de datos
# para manejar diferentes formatos de entrada podría verse así:

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Cargar modelo
model = joblib.load("models/dropout_predictor.pkl")
scaler = joblib.load("models/scaler.pkl")

# Definir esquema de entrada para la API
class PredictionInput(BaseModel):
    age_at_enrollment: int
    curricular_units_1st_sem_enrolled: int
    curricular_units_1st_sem_approved: int
    curricular_units_2nd_sem_enrolled: int
    curricular_units_2nd_sem_approved: int
    unemployment_rate: float

# Endpoint para la API
@app.post("/api/predict")
def predict_from_api(input_data: PredictionInput):
    # Convertir datos de la API al formato del modelo
    features = {
        "Age at enrollment": input_data.age_at_enrollment,
        "Curricular units 1st sem (enrolled)": input_data.curricular_units_1st_sem_enrolled,
        "Curricular units 1st sem (approved)": input_data.curricular_units_1st_sem_approved,
        "Curricular units 2nd sem (enrolled)": input_data.curricular_units_2nd_sem_enrolled,
        "Curricular units 2nd sem (approved)": input_data.curricular_units_2nd_sem_approved,
        "Unemployment rate": input_data.unemployment_rate
    }

    # Convertir a DataFrame
    features_df = pd.DataFrame([features])

    # Escalar características
    features_scaled = scaler.transform(features_df)

    # Hacer predicción
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

# Función para predecir desde un archivo CSV
def predict_from_csv(csv_path):
    # Leer datos del CSV
    data = pd.read_csv(csv_path)

    # Renombrar columnas si es necesario
    if "edad" in data.columns:
        data.rename(columns={"edad": "Age at enrollment"}, inplace=True)
    if "matriculados_sem1" in data.columns:
        data.rename(columns={"matriculados_sem1": "Curricular units 1st sem (enrolled)"}, inplace=True)
    if "aprobados_sem1" in data.columns:
        data.rename(columns={"aprobados_sem1": "Curricular units 1st sem (approved)"}, inplace=True)
    if "matriculados_sem2" in data.columns:
        data.rename(columns={"matriculados_sem2": "Curricular units 2nd sem (enrolled)"}, inplace=True)
    if "aprobados_sem2" in data.columns:
        data.rename(columns={"aprobados_sem2": "Curricular units 2nd sem (approved)"}, inplace=True)
    if "desempleo" in data.columns:
        data.rename(columns={"desempleo": "Unemployment rate"}, inplace=True)

    # Verificar que todas las columnas necesarias estén presentes
    required_columns = [
        "Age at enrollment", "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (approved)", "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (approved)", "Unemployment rate"
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas en el CSV: {missing_columns}")

    # Seleccionar solo las columnas necesarias
    data = data[required_columns]

    # Escalar características
    features_scaled = scaler.transform(data)

    # Hacer predicciones
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[:, 1]

    # Añadir predicciones y probabilidades al DataFrame
    data["prediction"] = predictions
    data["probability"] = probabilities

    return data

# Función para predecir desde un diccionario genérico
def predict_from_dict(data_dict):
    # Mapear claves del diccionario a columnas esperadas por el modelo
    features = {}

    # Edad puede estar en diferentes formatos
    if "age_at_enrollment" in data_dict:
        features["Age at enrollment"] = data_dict["age_at_enrollment"]
    elif "age" in data_dict:
        features["Age at enrollment"] = data_dict["age"]
    elif "enrollment_age" in data_dict:
        features["Age at enrollment"] = data_dict["enrollment_age"]
    else:
        raise ValueError("No se encontró información sobre la edad")

    # Unidades curriculares 1er semestre (matriculadas)
    if "curricular_units_1st_sem_enrolled" in data_dict:
        features["Curricular units 1st sem (enrolled)"] = data_dict["curricular_units_1st_sem_enrolled"]
    elif "enrolled_1" in data_dict:
        features["Curricular units 1st sem (enrolled)"] = data_dict["enrolled_1"]
    elif "units_enrolled_sem1" in data_dict:
        features["Curricular units 1st sem (enrolled)"] = data_dict["units_enrolled_sem1"]
    else:
        raise ValueError("No se encontró información sobre unidades matriculadas en 1er semestre")

    # Unidades curriculares 1er semestre (aprobadas)
    if "curricular_units_1st_sem_approved" in data_dict:
        features["Curricular units 1st sem (approved)"] = data_dict["curricular_units_1st_sem_approved"]
    elif "approved_1" in data_dict:
        features["Curricular units 1st sem (approved)"] = data_dict["approved_1"]
    elif "units_passed_sem1" in data_dict:
        features["Curricular units 1st sem (approved)"] = data_dict["units_passed_sem1"]
    else:
        raise ValueError("No se encontró información sobre unidades aprobadas en 1er semestre")

    # Unidades curriculares 2do semestre (matriculadas)
    if "curricular_units_2nd_sem_enrolled" in data_dict:
        features["Curricular units 2nd sem (enrolled)"] = data_dict["curricular_units_2nd_sem_enrolled"]
    elif "enrolled_2" in data_dict:
        features["Curricular units 2nd sem (enrolled)"] = data_dict["enrolled_2"]
    elif "units_enrolled_sem2" in data_dict:
        features["Curricular units 2nd sem (enrolled)"] = data_dict["units_enrolled_sem2"]
    else:
        raise ValueError("No se encontró información sobre unidades matriculadas en 2do semestre")

    # Unidades curriculares 2do semestre (aprobadas)
    if "curricular_units_2nd_sem_approved" in data_dict:
        features["Curricular units 2nd sem (approved)"] = data_dict["curricular_units_2nd_sem_approved"]
    elif "approved_2" in data_dict:
        features["Curricular units 2nd sem (approved)"] = data_dict["approved_2"]
    elif "units_passed_sem2" in data_dict:
        features["Curricular units 2nd sem (approved)"] = data_dict["units_passed_sem2"]
    else:
        raise ValueError("No se encontró información sobre unidades aprobadas en 2do semestre")

    # Tasa de desempleo
    if "unemployment_rate" in data_dict:
        features["Unemployment rate"] = data_dict["unemployment_rate"]
    elif "unemployment" in data_dict:
        features["Unemployment rate"] = data_dict["unemployment"]
    elif "jobless_rate" in data_dict:
        features["Unemployment rate"] = data_dict["jobless_rate"]
    else:
        raise ValueError("No se encontró información sobre la tasa de desempleo")

    # Convertir a DataFrame
    features_df = pd.DataFrame([features])

    # Escalar características
    features_scaled = scaler.transform(features_df)

    # Hacer predicción
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
"""

# -------------------------------------------------------------------------
# IMPLEMENTACIÓN CON PATRÓN ADAPTER
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import joblib
from pydantic import BaseModel
from abc import ABC, abstractmethod


# Definición de modelos Pydantic para la API
class PredictionInput(BaseModel):
    """Modelo Pydantic para la entrada de la API REST"""
    age_at_enrollment: int
    curricular_units_1st_sem_enrolled: int
    curricular_units_1st_sem_approved: int
    curricular_units_2nd_sem_enrolled: int
    curricular_units_2nd_sem_approved: int
    unemployment_rate: float


class PredictionOutput(BaseModel):
    """Modelo Pydantic para la salida de la API REST"""
    prediction: int
    probability: float
    prediction_text: str


class InputAdapter(ABC):
    """
    Interfaz abstracta para todos los adaptadores de entrada.
    Define los métodos que deben implementar todos los adaptadores.
    """

    @abstractmethod
    def to_model_format(self) -> Dict[str, Any]:
        """
        Convierte los datos al formato esperado por el modelo.

        Returns:
            Dict[str, Any]: Diccionario con las características en el formato esperado por el modelo
        """
        pass

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convierte los datos a un DataFrame de pandas.

        Returns:
            pd.DataFrame: DataFrame con las características en el formato esperado por el modelo
        """
        pass


class ApiInputAdapter(InputAdapter):
    """
    Adaptador para convertir datos de la API al formato esperado por el modelo.
    """

    def __init__(self, input_data: PredictionInput):
        """
        Inicializa el adaptador con los datos de la API.

        Args:
            input_data (PredictionInput): Objeto Pydantic con los datos de entrada de la API
        """
        self.input_data = input_data

    def to_model_format(self) -> Dict[str, Any]:
        """
        Convierte los datos de la API al formato esperado por el modelo.

        Returns:
            Dict[str, Any]: Diccionario con las características en el formato esperado por el modelo
        """
        return {
            "Age at enrollment": self.input_data.age_at_enrollment,
            "Curricular units 1st sem (enrolled)": self.input_data.curricular_units_1st_sem_enrolled,
            "Curricular units 1st sem (approved)": self.input_data.curricular_units_1st_sem_approved,
            "Curricular units 2nd sem (enrolled)": self.input_data.curricular_units_2nd_sem_enrolled,
            "Curricular units 2nd sem (approved)": self.input_data.curricular_units_2nd_sem_approved,
            "Unemployment rate": self.input_data.unemployment_rate
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convierte los datos de la API a un DataFrame de pandas.

        Returns:
            pd.DataFrame: DataFrame con las características en el formato esperado por el modelo
        """
        return pd.DataFrame([self.to_model_format()])


class DictionaryInputAdapter(InputAdapter):
    """
    Adaptador para convertir datos de un diccionario genérico al formato esperado por el modelo.
    """

    def __init__(self, input_dict: Dict[str, Any]):
        """
        Inicializa el adaptador con los datos del diccionario.

        Args:
            input_dict (Dict[str, Any]): Diccionario con los datos de entrada
        """
        self.input_dict = input_dict

        # Mapeo de claves alternativas a las claves esperadas por el modelo
        self.key_mappings = {
            "Age at enrollment": ["age_at_enrollment", "age", "enrollment_age", "edad"],
            "Curricular units 1st sem (enrolled)": ["curricular_units_1st_sem_enrolled", "enrolled_1", "units_enrolled_sem1", "matriculados_sem1"],
            "Curricular units 1st sem (approved)": ["curricular_units_1st_sem_approved", "approved_1", "units_passed_sem1", "aprobados_sem1"],
            "Curricular units 2nd sem (enrolled)": ["curricular_units_2nd_sem_enrolled", "enrolled_2", "units_enrolled_sem2", "matriculados_sem2"],
            "Curricular units 2nd sem (approved)": ["curricular_units_2nd_sem_approved", "approved_2", "units_passed_sem2", "aprobados_sem2"],
            "Unemployment rate": ["unemployment_rate", "unemployment", "jobless_rate", "desempleo"]
        }

    def to_model_format(self) -> Dict[str, Any]:
        """
        Convierte los datos del diccionario al formato esperado por el modelo.

        Returns:
            Dict[str, Any]: Diccionario con las características en el formato esperado por el modelo

        Raises:
            ValueError: Si falta alguna característica necesaria
        """
        model_features = {}
        missing_features = []

        # Para cada característica esperada por el modelo
        for model_key, alternative_keys in self.key_mappings.items():
            # Buscar la primera clave alternativa que exista en el diccionario
            found = False
            for alt_key in alternative_keys:
                if alt_key in self.input_dict:
                    model_features[model_key] = self.input_dict[alt_key]
                    found = True
                    break

            # Si no se encontró ninguna clave alternativa para esta característica
            if not found:
                missing_features.append(model_key)

        # Si faltan características, lanzar una excepción
        if missing_features:
            raise ValueError(f"Faltan características necesarias: {', '.join(missing_features)}")

        return model_features

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convierte los datos del diccionario a un DataFrame de pandas.

        Returns:
            pd.DataFrame: DataFrame con las características en el formato esperado por el modelo
        """
        return pd.DataFrame([self.to_model_format()])


class CsvInputAdapter(InputAdapter):
    """
    Adaptador para convertir datos de un archivo CSV al formato esperado por el modelo.
    """

    def __init__(self, csv_path: str):
        """
        Inicializa el adaptador con los datos del archivo CSV.

        Args:
            csv_path (str): Ruta al archivo CSV
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        # Mapeo de nombres de columnas alternativos a los esperados por el modelo
        self.column_mappings = {
            "edad": "Age at enrollment",
            "age": "Age at enrollment",
            "enrollment_age": "Age at enrollment",

            "matriculados_sem1": "Curricular units 1st sem (enrolled)",
            "enrolled_1": "Curricular units 1st sem (enrolled)",
            "units_enrolled_sem1": "Curricular units 1st sem (enrolled)",

            "aprobados_sem1": "Curricular units 1st sem (approved)",
            "approved_1": "Curricular units 1st sem (approved)",
            "units_passed_sem1": "Curricular units 1st sem (approved)",

            "matriculados_sem2": "Curricular units 2nd sem (enrolled)",
            "enrolled_2": "Curricular units 2nd sem (enrolled)",
            "units_enrolled_sem2": "Curricular units 2nd sem (enrolled)",

            "aprobados_sem2": "Curricular units 2nd sem (approved)",
            "approved_2": "Curricular units 2nd sem (approved)",
            "units_passed_sem2": "Curricular units 2nd sem (approved)",

            "desempleo": "Unemployment rate",
            "unemployment": "Unemployment rate",
            "jobless_rate": "Unemployment rate"
        }

        # Renombrar columnas si es necesario
        self._rename_columns()

    def _rename_columns(self):
        """
        Renombra las columnas del DataFrame según el mapeo.
        """
        # Crear un diccionario de mapeo inverso (de nombres alternativos a nombres esperados)
        rename_dict = {}
        for alt_name, model_name in self.column_mappings.items():
            if alt_name in self.df.columns:
                rename_dict[alt_name] = model_name

        # Renombrar las columnas
        if rename_dict:
            self.df.rename(columns=rename_dict, inplace=True)

    def to_model_format(self) -> Dict[str, Any]:
        """
        Este método no es aplicable para CSV, que suele contener múltiples registros.
        """
        raise NotImplementedError("El método to_model_format() no es aplicable para archivos CSV. Use to_dataframe() en su lugar.")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame con las características en el formato esperado por el modelo.

        Returns:
            pd.DataFrame: DataFrame con las características en el formato esperado por el modelo

        Raises:
            ValueError: Si faltan columnas necesarias en el CSV
        """
        # Verificar que todas las columnas necesarias estén presentes
        required_columns = [
            "Age at enrollment",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (approved)",
            "Unemployment rate"
        ]

        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Faltan columnas en el CSV: {', '.join(missing_columns)}")

        # Seleccionar solo las columnas necesarias
        return self.df[required_columns]


class JsonInputAdapter(InputAdapter):
    """
    Adaptador para convertir datos de un archivo JSON o una lista de diccionarios
    al formato esperado por el modelo.
    """

    def __init__(self, json_data: Union[List[Dict[str, Any]], str]):
        """
        Inicializa el adaptador con los datos JSON.

        Args:
            json_data: Puede ser una lista de diccionarios o una ruta a un archivo JSON
        """
        if isinstance(json_data, str):
            # Si es una ruta a un archivo JSON, leerlo
            self.df = pd.read_json(json_data)
        else:
            # Si es una lista de diccionarios, convertirla a DataFrame
            self.df = pd.DataFrame(json_data)

        # Crear un adaptador de diccionario para cada fila
        self.dict_adapters = []
        for _, row in self.df.iterrows():
            self.dict_adapters.append(DictionaryInputAdapter(row.to_dict()))

    def to_model_format(self) -> Dict[str, Any]:
        """
        Este método no es aplicable para JSON con múltiples registros.
        """
        if len(self.dict_adapters) > 1:
            raise NotImplementedError(
                "El método to_model_format() no es aplicable para JSON con múltiples registros. "
                "Use to_dataframe() en su lugar."
            )
        return self.dict_adapters[0].to_model_format()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame con las características en el formato esperado por el modelo.

        Returns:
            pd.DataFrame: DataFrame con las características en el formato esperado por el modelo
        """
        # Convertir cada diccionario a formato modelo y luego a DataFrame
        model_dicts = []
        for adapter in self.dict_adapters:
            try:
                model_dicts.append(adapter.to_model_format())
            except ValueError as e:
                # Ignorar filas con datos faltantes
                print(f"Se omitió una fila debido a: {str(e)}")

        return pd.DataFrame(model_dicts)


class InputAdapterFactory:
    """
    Fábrica para crear adaptadores de entrada según el tipo de datos.

    Combina el patrón Factory con el patrón Adapter para proporcionar
    un punto único de creación de adaptadores.
    """

    @staticmethod
    def create_adapter(input_data: Any) -> InputAdapter:
        """
        Crea un adaptador de entrada según el tipo de datos.

        Args:
            input_data: Datos de entrada (puede ser un objeto Pydantic, un diccionario, una ruta a un CSV, etc.)

        Returns:
            InputAdapter: Adaptador apropiado para el tipo de datos

        Raises:
            ValueError: Si el tipo de datos no es soportado
        """
        # Si es un objeto Pydantic (PredictionInput)
        if isinstance(input_data, PredictionInput):
            return ApiInputAdapter(input_data)

        # Si es un diccionario
        elif isinstance(input_data, dict):
            return DictionaryInputAdapter(input_data)

        # Si es una ruta a un archivo
        elif isinstance(input_data, str):
            if input_data.endswith('.csv'):
                return CsvInputAdapter(input_data)
            elif input_data.endswith('.json'):
                return JsonInputAdapter(input_data)
            else:
                raise ValueError(f"Formato de archivo no soportado: {input_data}")

        # Si es una lista de diccionarios (datos JSON)
        elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
            return JsonInputAdapter(input_data)

        else:
            raise ValueError(f"Tipo de datos no soportado: {type(input_data)}")


class DropoutPredictor:
    """
    Clase que encapsula el modelo de predicción de deserción estudiantil.
    """

    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Inicializa el predictor con un modelo y un scaler.

        Args:
            model_path: Ruta al archivo del modelo
            scaler_path: Ruta al archivo del scaler
        """
        self.model = None
        self.scaler = None

        if model_path:
            self.model = joblib.load(model_path)

        if scaler_path:
            self.scaler = joblib.load(scaler_path)

    def predict(self, features: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Realiza predicciones para las características proporcionadas.

        Args:
            features: DataFrame con las características

        Returns:
            List[Dict[str, Any]]: Lista de diccionarios con las predicciones
        """
        if self.model is None or self.scaler is None:
            raise ValueError("El modelo y el scaler deben ser cargados antes de hacer predicciones")

        # Escalar características
        features_scaled = self.scaler.transform(features)

        # Hacer predicciones
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]

        # Crear lista de resultados
        results = []
        for i in range(len(predictions)):
            results.append({
                "prediction": int(predictions[i]),
                "probability": float(probabilities[i]),
                "prediction_text": "Deserción" if predictions[i] == 1 else "Graduación"
            })

        return results


# Ejemplo de uso del patrón Adapter
def demo_adapter_pattern():
    """Demostración del uso del patrón Adapter"""

    # Simulamos tener un modelo y un scaler cargados
    # En un caso real, usaríamos rutas a archivos reales
    predictor = DropoutPredictor(
        model_path="../../models/dropout_predictor.pkl",
        scaler_path="../../models/scaler.pkl"
    )

    print("Demostración del patrón Adapter para diferentes fuentes de datos\n")

    # 1. Datos de la API (objeto Pydantic)
    print("1. Procesando datos de la API (objeto Pydantic):")
    api_data = PredictionInput(
        age_at_enrollment=20,
        curricular_units_1st_sem_enrolled=6,
        curricular_units_1st_sem_approved=5,
        curricular_units_2nd_sem_enrolled=6,
        curricular_units_2nd_sem_approved=5,
        unemployment_rate=10.8
    )

    # Crear adaptador usando la fábrica
    api_adapter = InputAdapterFactory.create_adapter(api_data)

    # Convertir datos al formato del modelo
    df1 = api_adapter.to_dataframe()
    print(f"Datos convertidos a formato del modelo:\n{df1.to_string()}")

    # Hacer predicción
    try:
        results1 = predictor.predict(df1)
        print(f"Resultado de la predicción: {results1[0]}\n")
    except Exception as e:
        print(f"Error en la predicción: {str(e)}\n")

    # 2. Datos de un diccionario (formato alternativo)
    print("2. Procesando datos de un diccionario (formato alternativo):")
    dict_data = {
        "age": 22,
        "enrolled_1": 7,
        "approved_1": 6,
        "enrolled_2": 7,
        "approved_2": 5,
        "unemployment": 12.3
    }

    # Crear adaptador usando la fábrica
    dict_adapter = InputAdapterFactory.create_adapter(dict_data)

    # Convertir datos al formato del modelo
    df2 = dict_adapter.to_dataframe()
    print(f"Datos convertidos a formato del modelo:\n{df2.to_string()}")

    # Hacer predicción
    try:
        results2 = predictor.predict(df2)
        print(f"Resultado de la predicción: {results2[0]}\n")
    except Exception as e:
        print(f"Error en la predicción: {str(e)}\n")

    # 3. Datos de un diccionario (formato español)
    print("3. Procesando datos de un diccionario (formato en español):")
    spanish_dict = {
        "edad": 25,
        "matriculados_sem1": 5,
        "aprobados_sem1": 4,
        "matriculados_sem2": 6,
        "aprobados_sem2": 3,
        "desempleo": 15.2
    }

    # Crear adaptador usando la fábrica
    spanish_adapter = InputAdapterFactory.create_adapter(spanish_dict)

    # Convertir datos al formato del modelo
    df3 = spanish_adapter.to_dataframe()
    print(f"Datos convertidos a formato del modelo:\n{df3.to_string()}")

    # Hacer predicción
    try:
        results3 = predictor.predict(df3)
        print(f"Resultado de la predicción: {results3[0]}\n")
    except Exception as e:
        print(f"Error en la predicción: {str(e)}\n")

    # 4. Datos de una lista de diccionarios (múltiples registros)
    print("4. Procesando datos de una lista de diccionarios (múltiples registros):")
    json_data = [
        {
            "age_at_enrollment": 19,
            "curricular_units_1st_sem_enrolled": 6,
            "curricular_units_1st_sem_approved": 6,
            "curricular_units_2nd_sem_enrolled": 6,
            "curricular_units_2nd_sem_approved": 6,
            "unemployment_rate": 9.5
        },
        {
            "age_at_enrollment": 23,
            "curricular_units_1st_sem_enrolled": 5,
            "curricular_units_1st_sem_approved": 3,
            "curricular_units_2nd_sem_enrolled": 5,
            "curricular_units_2nd_sem_approved": 2,
            "unemployment_rate": 11.2
        }
    ]

    # Crear adaptador usando la fábrica
    json_adapter = InputAdapterFactory.create_adapter(json_data)

    # Convertir datos al formato del modelo
    df4 = json_adapter.to_dataframe()
    print(f"Datos convertidos a formato del modelo:\n{df4.to_string()}")

    # Hacer predicción
    try:
        results4 = predictor.predict(df4)
        print("Resultados de la predicción:")
        for i, result in enumerate(results4):
            print(f"Estudiante {i+1}: {result}")
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")


if __name__ == "__main__":
    demo_adapter_pattern()
