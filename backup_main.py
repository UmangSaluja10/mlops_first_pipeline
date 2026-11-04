import os

import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Iris Model API")

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:6001")
MODEL_URI = "models:/iris-classifier@champion"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    client = mlflow.MlflowClient()
    champion_model = client.get_model_version_by_alias(
        name="iris-classifier", alias="champion"
    )
    print("Loaded champion model from MLflow:", champion_model.version)
except Exception as e:
    raise RuntimeError(f"Failed to load model from MLflow: {e}")


class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def root():
    return {"message": "API is ready to serve..."}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_name": "iris-classifier",
        "model_version": champion_model.version,
    }


@app.post("/predict")
def predict(data: IrisRequest):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
        "class_name": ["setosa", "versicolor", "virginica"][prediction],
    }