import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression
from faker import Faker
import numpy as np
import yaml
from typing import Tuple
import logging
import joblib
import os
from sklearn.metrics import mean_squared_error, accuracy_score

# Configuración del logger
logging.basicConfig(
    filename='log.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Solo se mostrarán mensajes de INFO y superior
)

logger = logging.getLogger(__name__)

class Validator:
    """
    A utility class for validating various types of data inputs.
    Contains static methods for validating DataFrames, strings, numbers, columns, and functions.
    """

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, var_name: str = "df") -> None:
        logger.info(f"Validating if {var_name} is a pandas DataFrame.")
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Validation failed: {var_name} is not a pandas DataFrame.", exc_info=True)
            raise ValueError(f"{var_name} must be a pandas DataFrame.")
        logger.info(f"{var_name} validated as a pandas DataFrame.")

    @staticmethod
    def validate_string(value: str, var_name: str = "value") -> None:
        logger.info(f"Validating if {var_name} is a non-empty string.")
        if not isinstance(value, str):
            logger.error(f"Validation failed: {var_name} is not a string.", exc_info=True)
            raise ValueError(f"{var_name} must be a string.")
        if not value.strip():
            logger.error(f"Validation failed: {var_name} is an empty string.", exc_info=True)
            raise ValueError(f"{var_name} cannot be empty.")
        logger.info(f"{var_name} validated as a non-empty string.")

    @staticmethod
    def validate_number(value: int | float, var_name: str = "value", check_positive: bool = True) -> None:
        logger.info(f"Validating if {var_name} is a number.")
        
        # Verifica si el valor es un número
        if not isinstance(value, (int, float)):
            logger.error(f"Validation failed: {var_name} is not a number.", exc_info=True)
            raise ValueError(f"{var_name} must be a number (int or float).")
        
        # Verifica si el número es positivo (opcional, activado por defecto)
        if check_positive and value <= 0:
            logger.error(f"Validation failed: {var_name} must be a positive number.", exc_info=True)
            raise ValueError(f"{var_name} must be a positive number.")
        
        logger.info(f"{var_name} validated as a number.")

    @staticmethod
    def validate_column_exists(df: pd.DataFrame, column_name: str) -> None:
        logger.info(f"Validating if column {column_name} exists in the DataFrame.")
        Validator.validate_dataframe(df)
        Validator.validate_string(column_name, "column_name")
        
        if column_name not in df.columns:
            logger.error(f"Validation failed: Column '{column_name}' does not exist in the DataFrame.", exc_info=True)
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        logger.info(f"Column {column_name} exists in the DataFrame.")

    @staticmethod
    def validate_function(func) -> None:
        logger.info(f"Validating if {func} is a callable function.")
        if not callable(func):
            logger.error(f"Validation failed: {func} is not a valid function.", exc_info=True)
            raise ValueError(f"'{func}' must be a valid function.")
        logger.info(f"{func} validated as a callable function.")

    @staticmethod
    def validate_list(input_list, var_name="list"):
        logger.info(f"Validating if {var_name} is a list.")
        if not isinstance(input_list, list):
            logger.error(f"Validation failed: {var_name} is not a list.", exc_info=True)
            raise ValueError(f"{var_name} must be a list.")
        logger.info(f"{var_name} validated as a list.")
    
    @staticmethod
    def validate_boolean(value, var_name: str = "value") -> None:
        logger.info(f"Validating if {var_name} is a boolean.")
        
        if not isinstance(value, bool):
            logger.error(f"Validation failed: {var_name} is not a boolean.", exc_info=True)
            raise ValueError(f"{var_name} must be a boolean (True or False).")
        
        logger.info(f"{var_name} validated as a boolean.")    


class DataHandler:
    """
    Class for handling data manipulation tasks on pandas DataFrames.
    This class interacts with the Validator to ensure inputs are valid.
    """

    @staticmethod
    def filter_rows_by_threshold(df: pd.DataFrame, column_name: str, threshold: float) -> pd.DataFrame:
        logger.info(f"Calling filter_rows_by_threshold with column_name={column_name}, threshold={threshold}")

        try:
            Validator.validate_dataframe(df)
            Validator.validate_column_exists(df, column_name)
            Validator.validate_number(threshold, "threshold", False)

            filtered_df = df[df[column_name] > threshold]
            logger.info(f"Filtered DataFrame with rows where {column_name} > {threshold}.")
            return filtered_df
        except Exception as e:
            logger.error(f"Error filtering rows by threshold: {e}", exc_info=True)
            raise

    @staticmethod
    def group_and_calculate_mean(df: pd.DataFrame, group_column: str, agg_column: str) -> pd.DataFrame:
        logger.info(f"Calling group_and_calculate_mean with group_column={group_column}, agg_column={agg_column}")

        try:
            Validator.validate_dataframe(df)
            Validator.validate_column_exists(df, group_column)
            Validator.validate_column_exists(df, agg_column)

            grouped_df = df.groupby(group_column)[agg_column].mean().reset_index()
            logger.info(f"Grouped DataFrame by {group_column} and calculated the mean of {agg_column}.")
            return grouped_df
        except Exception as e:
            logger.error(f"Error grouping and calculating mean: {e}", exc_info=True)
            raise

    @staticmethod
    def apply_function_to_column(df: pd.DataFrame, column_name: str, func) -> pd.DataFrame:
        logger.info(f"Calling apply_function_to_column with column_name={column_name}, func={func}")

        try:
            Validator.validate_dataframe(df)
            Validator.validate_column_exists(df, column_name)
            Validator.validate_function(func)

            df[column_name] = df[column_name].apply(func)
            logger.info(f"Applied function to column {column_name}.")
            return df
        except Exception as e:
            logger.error(f"Error applying function to column: {e}", exc_info=True)
            raise        

class ModelBuilder:
    """
    Class for building and training regression and classification models, now utilizing configuration from a .yaml file.
    """

    @staticmethod
    def create_model_dir(directory: str = "./modelos"):
        """Creates the directory to store models if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Directory {directory} created.")
        else:
            logger.info(f"Directory {directory} already exists.")
    
    @staticmethod
    def load_config(config_file: str = "config.yaml"):
        logger.info(f"Loading configuration from {config_file}")
        try:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully.")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}", exc_info=True)
            raise

    @staticmethod
    def generate_regression_data(config: dict, n_samples: int = None, n_features: int = None, n_digits: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Calling generate_regression_data with n_samples={n_samples}, n_features={n_features}")

        # Validaciones
        n_samples = n_samples or config['regression_data']['n_samples']
        n_features = n_features or config['regression_data']['n_features']
        n_digits = n_digits or config['regression_data']['n_digits']

        Validator.validate_number(n_samples, "n_samples")
        Validator.validate_number(n_features, "n_features")
        Validator.validate_number(n_digits, "n_digits")

        try:
            fake = Faker()
            X = [[fake.random_number(digits=n_digits) for _ in range(n_features)] for _ in range(n_samples)]
            y = [fake.random_number(digits=n_digits) for _ in range(n_samples)]
            df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
            target = pd.Series(y, name="Target")
            
            logger.info(f"Generated regression data with {n_samples} samples and {n_features} features.")
            return df, target
        except Exception as e:
            logger.error(f"Error generating regression data: {e}", exc_info=True)
            raise

    @staticmethod
    def train_multiple_regression_model(df: pd.DataFrame, target: pd.Series, config: dict, fit_intercept: bool = None, save_model: bool = False) -> LinearRegression:
        logger.info(f"Calling train_multiple_regression_model with fit_intercept={fit_intercept} and save_model={save_model}")

        fit_intercept = fit_intercept if fit_intercept is not None else config['multiple_regression_model']['fit_intercept']

        # Validaciones
        Validator.validate_dataframe(df, "df")
        Validator.validate_list(target.tolist(), "target")
        Validator.validate_boolean(fit_intercept, "intercept")
        if df.empty or target.empty:
            logger.error("DataFrame or target is empty.", exc_info=True)
            raise ValueError("DataFrame or target is empty.")

        try:
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(df, target)

            # Log intercepto y coeficientes (pesos)
            logger.info(f"Intercept: {model.intercept_}")
            logger.info(f"Coefficients: {model.coef_}")

            # Calcular métricas importantes
            predictions = model.predict(df)
            mse = mean_squared_error(target, predictions)
            logger.info(f"Mean Squared Error: {mse}")

            # Guardar el modelo si se indica
            if save_model:
                ModelBuilder.create_model_dir()  # Crear la carpeta si no existe
                model_path = os.path.join("modelos", "multiple_regression_model.pkl")
                joblib.dump(model, model_path)
                logger.info(f"Multiple regression model saved at {model_path}")

            return model
        except Exception as e:
            logger.error(f"Error training multiple regression model: {e}", exc_info=True)
            raise

    @staticmethod
    def train_logistic_regression_model(X: pd.DataFrame, y: pd.Series, config: dict, penalty: str = None, solver: str = None, C: float = None, save_model: bool = False) -> LogisticRegression:
        logger.info(f"Calling train_logistic_regression_model with penalty={penalty}, solver={solver}, C={C}, and save_model={save_model}")

        # Validaciones
        Validator.validate_dataframe(X, "X")
        Validator.validate_list(y.tolist(), "y")
        if X.empty or y.empty:
            raise ValueError("DataFrame or target is empty.")
        if len(np.unique(y)) != 2:
            logger.error("Error: The target variable (y) must be binary.", exc_info=True)
            raise ValueError("The target variable (y) must be binary for logistic regression.")

        penalty = penalty or config['logistic_regression_model']['penalty']
        solver = solver or config['logistic_regression_model']['solver']
        C = C or config['logistic_regression_model']['C']

        Validator.validate_string(penalty, "penalty")
        Validator.validate_string(solver, "solver")
        Validator.validate_number(C, "C")

        try:
            model = LogisticRegression(penalty=penalty, solver=solver, C=C)
            model.fit(X, y)

            # Log intercepto y coeficientes (pesos)
            logger.info(f"Intercept: {model.intercept_}")
            logger.info(f"Coefficients: {model.coef_}")

            # Calcular métricas importantes
            predictions = model.predict(X)
            accuracy = accuracy_score(y, predictions)
            logger.info(f"Accuracy: {accuracy}")

            # Guardar el modelo si se indica
            if save_model:
                ModelBuilder.create_model_dir()  # Crear la carpeta si no existe
                model_path = os.path.join("modelos", "logistic_regression_model.pkl")
                joblib.dump(model, model_path)
                logger.info(f"Logistic regression model saved at {model_path}")

            return model
        except Exception as e:
            logger.error(f"Error training logistic regression model: {e}", exc_info=True)
            raise

    @staticmethod
    def generate_logistic_regression_data(config: dict, n_samples: int = None, n_features: int = None, n_digits: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Calling generate_logistic_regression_data with n_samples={n_samples}, n_features={n_features}")

        # Validaciones
        n_samples = n_samples or config['regression_data']['n_samples']
        n_features = n_features or config['regression_data']['n_features']
        n_digits = n_digits or config['regression_data']['n_digits']

        Validator.validate_number(n_samples, "n_samples")
        Validator.validate_number(n_features, "n_features")
        Validator.validate_number(n_digits, "n_digits")

        try:
            fake = Faker()
            X = [[fake.random_number(digits=n_digits) for _ in range(n_features)] for _ in range(n_samples)]
            y = np.random.choice([0, 1], size=n_samples)
            df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
            target = pd.Series(y, name="Target")

            logger.info(f"Generated logistic regression data with {n_samples} samples and {n_features} features.")
            return df, target
        except Exception as e:
            logger.error(f"Error generating logistic regression data: {e}", exc_info=True)
            raise


class DataUtils:
    """
    Utility class for performing operations like flattening and filtering lists with comprehensions.
    """

    @staticmethod
    def flatten_list(nested_list):
        logger.info(f"Calling flatten_list with nested_list={nested_list}")

        try:
            Validator.validate_list(nested_list, "nested_list")
            for sublist in nested_list:
                Validator.validate_list(sublist, "sublist inside nested_list")

            flat_list = [item for sublist in nested_list for item in sublist]
            logger.info("Nested list flattened successfully.")
            return flat_list
        except Exception as e:
            logger.error(f"Error flattening the list: {e}", exc_info=True)
            raise

    @staticmethod
    def filter_and_square(numbers):
        logger.info(f"Calling filter_and_square with numbers={numbers}")

        try:
            Validator.validate_list(numbers, "numbers")
            for number in numbers:
                Validator.validate_number(number, "element in numbers",False)

            squared = [x ** 2 for x in numbers if x > 5]
            logger.info("Numbers filtered and squared successfully.")
            return squared
        except Exception as e:
            logger.error(f"Error filtering and squaring the numbers: {e}", exc_info=True)
            raise        