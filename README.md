# Intermediate Level Exam

## Table of Contents
- Description
- Project Structure
- Installation
- Running the Project and Usage Examples
- API Endpoints
- Logs
- Authors

# Description

## The exam includes the resolution of the following exercises:

1. Filter DataFrame with pandas: Implement a function that receives a DataFrame, a column, and a threshold, and returns a filtered DataFrame.
2. Generate regression data: Use the Faker library to generate simulated data for a regression problem.
3. Train multiple regression model: Train a linear regression model using scikit-learn.
4. Nested list comprehension: Flatten a list of lists using list comprehensions.
5. Group and aggregate with pandas: Group data in a DataFrame by a column and calculate the mean.
6. Logistic classification model: Train a logistic regression model with binary data.
7. Apply function to a column: Apply a custom function to a DataFrame column.
8. Comprehensions with conditions: Filter and square numbers greater than 5 from a list using list comprehensions.

These problems have been grouped into three classes DataHandler, ModelBuilder and DataUtils. Additionally, an API has been created that allows these functions to be executed through HTTP requests, facilitating their use in various applications.

# Project Structure

```
.
├── api_service.py               # API service with FastAPI
├── config.yaml                  # Configuration file with default parameters
├── dockerfile                   # Dockerfile to create the Docker image
├── exam.py                      # Main script with all implemented functions
├── requirements.txt             # File with necessary dependencies
└── example_petitions.ipynb     # Jupyter Notebook with examples for each exercise
```

# Installation

## Installing Dependencies

Clone this repository and create a virtual environment to install the necessary dependencies:

```
git clone https://github.com/your-username/intermediate_level_exam
cd intermediate_level_exam
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

# Running the Project and Usage Examples


## Run Locally

- Activate the virtual environment:

```
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
- Run directly calling the functions of "examen.py":

```
python -c "
from examen import DataUtils
nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
flattened = DataUtils.flatten_list(nested_list)
print(flattened)"
```
You can also found diferent examples for each excercise in "Example_requests.ipynb",or also you can start API service.

- Start the API service:

```
uvicorn api_service:app --reload
```

The API will be available at http://localhost:8000 and you can access the interactive documentation at http://localhost:8000/docs or use examples in "Example_requests.ipynb".

## Requests with curl

To make a request to the endpoint that generates regression data:

```
curl -X POST "http://localhost:8000/generate_regression_data" \
-H "Content-Type: application/json" \
-d '{"n_samples": 100, "n_features": 3, "n_digits": 5}'
```
## Examples with Partial Data

If you don't specify all parameters in a request, the API will take the default values from the config.yaml file:

```
curl -X POST "http://localhost:8000/generate_regression_data" \
-H "Content-Type: application/json" \
-d '{}'
```

In this case, the data will be generated according to the parameters defined in config.yaml.

You can see all examples on Example_requests.ipynb, where each exercise has been implemented and tested independently.

## Run in Docker

- Build the Docker image:

```
docker build -t intermediate_level_exam .
```

- Run and interact with the container:

```
docker run -it -d -p 8000:8000 intermediate_level_exam
docker exec -it intermediate_level_exam
```

With this you can interact with the container to run python scrips or run API with uvicorn, if is used API you can send request at http://localhost:8000 locally and inside the container.

# API Endpoints

- POST /generate_regression_data: Generates simulated data for regression
- POST /train_multiple_regression: Trains a multiple regression model
- POST /train_logistic_regression: Trains a logistic regression model
- POST /filter_rows_by_threshold: Filters a DataFrame by a column and threshold
- POST /group_and_calculate_mean: Groups a DataFrame and calculates the mean
- POST /apply_function_to_column: Applies a function to a DataFrame column
- POST /flatten_list: Flattens a list of lists
- POST /filter_and_square: Filters and squares numbers greater than 5
- POST /predict_regression : Make predictions for regression model
- POST /predict_logistic_regression : Make predictions for logistic regression model

# Logs

The system generates a log file (log.log) that records each operation performed through the API, including details about errors and failed validations.

# Authors

- Luis Donaldo Romero Tapia
