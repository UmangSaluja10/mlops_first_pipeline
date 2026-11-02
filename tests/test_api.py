# TestClient lets us send simulated HTTP requests
from fastapi.testclient import TestClient

# Import the application object defined in main.py
from main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify the app is running and responds to a health check."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_iris_prediction():
    """Verify that we get a valid prediction."""
    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert "prediction" in result
    assert "class_name" in result
    assert result["class_name"] in ["setosa", "versicolor", "virginica"]
    assert result["prediction"] in [0, 1, 2]