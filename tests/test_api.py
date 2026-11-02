#import testclient so we can send simulated HTTP requests to our API
from unittest import result

from fastapi.testclient import TestClient

# import application instance from main.py
from main import app

#reusable test client for sending requests to the API
client = TestClient(app)

def test_health_endpoint():
    #verify that the application is running and can respond to requests
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    
def test_valid_iris_prediction():

    # verify that we are getting a valid prediction for a valid input
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    # send a  simulated POST request  with json input to the /predict endpoint with the payload
    response = client.post("/predict", json=payload)

    # end point should accept a valid input
    assert response.status_code == 200

    # verify that the response contains a valid prediction and class name

    response_data = response.json()

    assert "prediction" in response_data
    assert "class_name" in response_data

    assert response_data["class_name"] in ["setosa", "versicolor", "virginica"]
    assert response_data["prediction"] in [0, 1, 2]