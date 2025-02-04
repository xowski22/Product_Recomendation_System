from locust import HttpUser, task, between

class LoadTest(HttpUser):
    wait_time = between(1, 2)
    host = "http://localhost:8080/"

    @task(1)
    def test_predict_rating(self):
        self.client.post(
            "/predict/rating/",
            json={"user_id": "1", "item_id": "1"}
        )

    @task(2)
    def test_recomendations(self):
        self.client.post(
            "/recommend/",
            json={"user_id": "1", "n_recommendations": 5}
        )