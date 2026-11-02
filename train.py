from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import mlflow
import mlflow.sklearn
# Client API lets us manage registered models and their aliases
from mlflow import MlflowClient

# Configure MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:6001")

# Group related training runs under one experiment
mlflow.set_experiment("iris-classification")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

n_estimators = 10
random_state = 42

# Model settings we want to track in MLflow
model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.20,
}

# Start a tracked training run
with mlflow.start_run() as run:
    mlflow.log_params(model_parameters)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # Model quality gate
    MIN_TEST_ACCURACY = 0.90
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(
            f"Model rejected: test accuracy {accuracy:.2f} is below {MIN_TEST_ACCURACY}"
        )

    print("Model Quality Gate Passed. Model is eligible for registration")

    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
    })

    print("MLflow run ID:", run.info.run_id)
    print("Test accuracy:", accuracy)
    print("Test precision:", precision)
    print("Test recall:", recall)

    # Save the trained model as a reusable artifact
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "iris_model.joblib")
    print(f"Model saved to {model_dir / 'iris_model.joblib'}")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris-model",
        registered_model_name="iris-classifier",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ],
    )

    # Assign the "champion" alias to the newly registered model
    client = MlflowClient()
    current_run_id = mlflow.active_run().info.run_id
    registered_versions = client.search_model_versions("name='iris-classifier'")
    current_model_version = next(
        (v for v in registered_versions if v.run_id == current_run_id),
        None,
    )

    if current_model_version is None:
        raise RuntimeError(
            "Model was logged but its registered version could not be found"
        )

    client.set_registered_model_alias(
        name="iris-classifier",
        alias="champion",
        version=current_model_version.version,
    )

    print(f"Champion alias assigned to model version {current_model_version.version}")
    print("Registered Model: iris-classifier")