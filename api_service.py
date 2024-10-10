from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import joblib
import logging
from examen import ModelBuilder, DataHandler, DataUtils, Validator

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Diccionario de operaciones seguras
operations_map = {
    "*2": lambda x: x * 2,
    "+2": lambda x: x + 2,
    "/2": lambda x: x / 2,
    "-2": lambda x: x - 2,
    "square": lambda x: x ** 2,
    "sqrt": lambda x: x ** 0.5,
    "exp": lambda x: x ** x
}

# Modelos para los distintos inputs
class DataFrameInput(BaseModel):
    columns: Dict[str, List]

class RegressionDataInput(BaseModel):
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    n_digits: Optional[int] = None

class TrainModelInput(DataFrameInput):
    target: List[float]
    fit_intercept: Optional[bool] = None
    save_model: Optional[bool] = False

class LogisticTrainInput(DataFrameInput):
    y: List[int]
    penalty: Optional[str] = None
    solver: Optional[str] = None
    C: Optional[float] = None
    save_model: Optional[bool] = False

class PredictionInput(DataFrameInput):
    model_name: str

class DataFilterInput(DataFrameInput):
    column_name: str
    threshold: float

class GroupAggInput(DataFrameInput):
    group_column: str
    agg_column: str

class ApplyFunctionInput(DataFrameInput):
    column_name: str
    operation: str

class ListProcessInput(BaseModel):
    input_list: List[List[int]]

class NumberListInput(BaseModel):
    numbers: List[int]    

# Función para convertir la entrada en formato dict a DataFrame de pandas
def dict_to_dataframe(data: Dict[str, List]) -> pd.DataFrame:
    try:
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logger.error(f"Error converting data to DataFrame: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error converting data to DataFrame: {e}")

# Rutas de la API

@app.post("/generate_regression_data")
async def generate_regression_data(input_data: RegressionDataInput):
    logger.info("API Request: Generate regression data.")
    try:
        df, target = ModelBuilder.generate_regression_data(
            ModelBuilder.load_config(),
            input_data.n_samples,
            input_data.n_features,
            input_data.n_digits
        )
        logger.info("API Response: Regression data generated successfully.")
        return {"features": df.to_dict(), "target": target.tolist()}
    except Exception as e:
        logger.error(f"API Error: Failed to generate regression data: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train_multiple_regression")
async def train_multiple_regression(input_data: TrainModelInput):
    logger.info("API Request: Train multiple regression model.")
    try:
        df = dict_to_dataframe(input_data.columns)
        target = pd.Series(input_data.target)
        model = ModelBuilder.train_multiple_regression_model(
            df, target, ModelBuilder.load_config(),
            input_data.fit_intercept, input_data.save_model
        )
        logger.info("API Response: Multiple regression model trained successfully.")
        return {"message": "Model trained successfully"}
    except Exception as e:
        logger.error(f"API Error: Failed to train multiple regression model: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train_logistic_regression")
async def train_logistic_regression(input_data: LogisticTrainInput):
    logger.info("API Request: Train logistic regression model.")
    try:
        X = dict_to_dataframe(input_data.columns)
        y = pd.Series(input_data.y)
        model = ModelBuilder.train_logistic_regression_model(
            X, y, ModelBuilder.load_config(),
            input_data.penalty, input_data.solver,
            input_data.C, input_data.save_model
        )
        logger.info("API Response: Logistic regression model trained successfully.")
        return {"message": "Logistic regression model trained successfully"}
    except Exception as e:
        logger.error(f"API Error: Failed to train logistic regression model: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

# Nueva ruta para predicción con un modelo de regresión
@app.post("/predict_regression")
async def predict_regression(input_data: PredictionInput):
    logger.info(f"API Request: Predict using regression model {input_data.model_name}.")
    try:
        df = dict_to_dataframe(input_data.columns)
        model_path = f"./modelos/{input_data.model_name}.pkl"
        model = joblib.load(model_path)
        predictions = model.predict(df)
        logger.info(f"API Response: Predictions made successfully with model {input_data.model_name}.")
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"API Error: Failed to make predictions with model {input_data.model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

# Nueva ruta para predicción con un modelo de regresión logística
@app.post("/predict_logistic_regression")
async def predict_logistic_regression(input_data: PredictionInput):
    logger.info(f"API Request: Predict using logistic regression model {input_data.model_name}.")
    try:
        df = dict_to_dataframe(input_data.columns)
        model_path = f"./modelos/{input_data.model_name}.pkl"
        model = joblib.load(model_path)
        predictions = model.predict(df)
        logger.info(f"API Response: Predictions made successfully with logistic model {input_data.model_name}.")
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"API Error: Failed to make predictions with logistic model {input_data.model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/filter_rows_by_threshold")
async def filter_rows_by_threshold(input_data: DataFilterInput):
    logger.info("API Request: Filter rows by threshold.")
    try:
        df = dict_to_dataframe(input_data.columns)
        filtered_df = DataHandler.filter_rows_by_threshold(df, input_data.column_name, input_data.threshold)
        logger.info("API Response: Rows filtered successfully.")
        return {"filtered_data": filtered_df.to_dict()}
    except Exception as e:
        logger.error(f"API Error: Failed to filter rows: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/group_and_calculate_mean")
async def group_and_calculate_mean(input_data: GroupAggInput):
    logger.info("API Request: Group and calculate mean.")
    try:
        df = dict_to_dataframe(input_data.columns)
        result = DataHandler.group_and_calculate_mean(df, input_data.group_column, input_data.agg_column)
        logger.info("API Response: Grouping and mean calculation successful.")
        return {"result": result.to_dict()}
    except Exception as e:
        logger.error(f"API Error: Failed to group and calculate mean: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/apply_function_to_column")
async def apply_function_to_column(input_data: ApplyFunctionInput):
    logger.info(f"API Request: Apply operation {input_data.operation} to column {input_data.column_name}.")
    try:
        df = dict_to_dataframe(input_data.columns)
        if input_data.operation not in operations_map:
            raise HTTPException(status_code=400, detail="Operation not supported")
        
        operation = operations_map[input_data.operation]
        modified_df = DataHandler.apply_function_to_column(df, input_data.column_name, operation)
        logger.info(f"API Response: Operation {input_data.operation} applied successfully to column {input_data.column_name}.")
        return {"modified_data": modified_df.to_dict()}
    except Exception as e:
        logger.error(f"API Error: Failed to apply operation to column: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/flatten_list")
async def flatten_list(input_data: ListProcessInput):
    try:
        flattened_list = DataUtils.flatten_list(input_data.input_list)
        return {"flattened_list": flattened_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/filter_and_square")
async def filter_and_square(input_data: NumberListInput):
    try:
        squared_list = DataUtils.filter_and_square(input_data.numbers)
        return {"squared_list": squared_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate_logistic_regression_data")
async def generate_logistic_regression_data(input_data: RegressionDataInput):
    try:
        df, target = ModelBuilder.generate_logistic_regression_data(
            ModelBuilder.load_config(),
            input_data.n_samples,
            input_data.n_features,
            input_data.n_digits
        )
        return {"features": df.to_dict(), "target": target.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


