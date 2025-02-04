import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import torch

from src.api.app import app

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

@pytest.fixture
def mock_mappings():
    user_mapping = {1: 0, 2: 1, 3:2}
    item_mapping = {1: 0, 2: 1, 3: 2}
    return user_mapping, item_mapping

@pytest.fixture
def mock_model():
    model = Mock()
    model.eval = Mock(return_value=None)
    def side_effect(user_tensor, item_tensor):
        if len(item_tensor) == 1:
            return torch.tensor([0.7])
        else:
            return torch.tensor([0.7, 0.6, 0.5])

    model.side_effect = side_effect
    model.__call__ = Mock(side_effect=side_effect)
    model.weight = torch.nn.Parameter(torch.randn(1))

    return model

class TestPredictRating:
    def test_valid_prediction(self, client, mock_mappings, mock_model):
        with patch('src.api.app.model', mock_model), \
            patch('src.api.app.user_mapping', mock_mappings[0]), \
            patch('src.api.app.item_mapping', mock_mappings[1]):

            response = client.post(
                "/predict/rating/",
                json={"user_id": "1", "item_id": "2"}
            )

            assert response.status_code == 200
            assert "predicted_rating" in response.json()
            assert isinstance(response.json()["predicted_rating"], float)

    def test_invalid_user_id(self, client, mock_mappings, mock_model):
        with patch('src.api.app.model', mock_model), \
            patch('src.api.app.user_mapping', mock_mappings[0]), \
            patch('src.api.app.item_mapping', mock_mappings[1]):

            response = client.post(
                "/predict/rating",
                json={"user_id": "999", "item_id": "1"}
            )

            assert response.status_code == 404
            assert "User mapping not found" in response.json()["detail"]


    def test_invalid_input_format(self, client, mock_mappings, mock_model):
        with patch('src.api.app.model', mock_model), \
                patch('src.api.app.user_mapping', mock_mappings[0]), \
                patch('src.api.app.item_mapping', mock_mappings[1]):

            response = client.post(
                "/recommend/",
                json={"user_id": "1", "n_recommendations": 101}
            )

            assert response.status_code == 422

def test_health_check(client, mock_mappings, mock_model):
    with patch('src.api.app.model', mock_model), \
            patch('src.api.app.user_mapping', mock_mappings[0]), \
            patch('src.api.app.item_mapping', mock_mappings[1]):

        response = client.get("/health/")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] == True
        
class TestErrorHandling:
    def test_model_not_loaded(self, client):
        with patch('src.api.app.model', None):
            response = client.post(
                "/predict/rating",
                json={"user_id": "1", "item_id": "1"}
            )
            assert response.status_code == 503

    def test_invalid_json(self, client):
            response = client.post(
                "/predict/rating",
                data="invalid json"
            )
            assert response.status_code == 422


def test_full_flow(client, mock_mappings, mock_model):
    with patch('src.api.app.model', mock_model), \
            patch('src.api.app.user_mapping', mock_mappings[0]), \
            patch('src.api.app.item_mapping', mock_mappings[1]):

        health_response = client.get("/health/")
        assert health_response.status_code == 200

        rating_response = client.post(
            "/predict/rating/",
            json={"user_id": "1", "item_id": "1"}
        )
        assert rating_response.status_code == 200

        print(f"Rating response: {rating_response.json() if rating_response.status_code == 200 else rating_response.text}")

        recommend_response = client.post(
            "/recommend/",
            json={"user_id": "1", "n_recommendations": 5}
        )
        print(f"Recommed response: {recommend_response.json() if recommend_response.status_code == 200 else recommend_response.text}")

        if recommend_response.status_code == 200:
            data = recommend_response.json()
            assert "items" in data
            assert "scores" in data
            assert all(isinstance(item, str) for item in data["items"])

        assert recommend_response.status_code == 200

class TestParameterValidation:
    @pytest.mark.parametrize("user_id,item_id,expected_status", [
        ("1", "2", 200),
        ("abc", "1", 400),
        ("1", "abc", 400),
        ("", "1", 400),
        ("1", "", 400),
    ])

    def test_rating_parameters(self, client, mock_mappings, mock_model, user_id, item_id, expected_status):
        with patch('src.api.app.model', mock_model), \
                patch('src.api.app.user_mapping', mock_mappings[0]), \
                patch('src.api.app.item_mapping', mock_mappings[1]):
            response = client.post(
                "/predict/rating",
                json={"user_id": user_id, "item_id": item_id}
            )
            assert response.status_code == expected_status

    @pytest.mark.parametrize("n_recommendations,expected_status", [
        (1, 200),
        (100, 200),
        (101, 422),
        (0, 422),
        (-1, 422),
    ])

    def test_recommendations_parameters(self, client, mock_mappings, mock_model, n_recommendations, expected_status):
        with patch('src.api.app.model', mock_model), \
                patch('src.api.app.user_mapping', mock_mappings[0]), \
                patch('src.api.app.item_mapping', mock_mappings[1]):
            response = client.post(
                "/recommend/",
                json={"user_id": "1", "n_recommendations": n_recommendations}
            )
            if response.status_code == expected_status:
                print(f"Recommendations response: {response.json() if response.status_code == 200 else response.text}")
            assert response.status_code == expected_status