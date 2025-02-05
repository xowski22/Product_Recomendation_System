from locust import HttpUser, task, between, events
from typing import Dict, Any
import json
import time
import random

class RecommendationSystemUser(HttpUser):
    wait_time = between(1, 5)
    host = "http://localhost:8080/"

    def on_start(self):
        self.user_ids = list(range(1, 6041))
        self.item_ids = list(range(1, 3953))

        self.rec_sizes = [5, 10, 20, 50, 100]

    def log_response_time(self, response_time: float, name: str, extra_data: Dict[str, Any] = None):
        if extra_data is None:
            extra_data = {}

        events.request.fire(
            request_type="CUSTOM",
            name=name,
            response_time=response_time,
            response_length=0,
            exception=None,
            context=extra_data
        )

    @task(3)
    def test_single_rating_prediction(self):
        user_id = str(random.choice(self.user_ids))
        item_id = str(random.choice(self.item_ids))

        start_time = time.time()

        self.client.post(
            "/predict/rating/",
            json={"user_id": user_id, "item_id": item_id}
        )

        response_time = (time.time() - start_time) * 1000

        self.log_response_time(
            response_time,
            "Single Rating Prediction",
            {"user_id": user_id, "item_id": item_id}
        )

    @task(2)
    def test_recomendations_varying_sizes(self):
        user_id = str(random.choice(self.user_ids))
        n_recommendations = random.choice(self.rec_sizes)

        start_time = time.time()

        self.client.post(
            "/recommend/",
            json={"user_id": user_id, "n_recommendations": n_recommendations}
        )

        response_time = (time.time() - start_time) * 1000

        self.log_response_time(
            response_time,
            f"Recommendations (n={n_recommendations})",
            {"user_id": user_id, "n_recommendations": n_recommendations}
        )

    @task(1)
    def test_batch_rating_prediction(self):
        user_id = str(random.choice(self.user_ids))
        batch_size = random.choice([5, 10, 20])

        start_time = time.time()

        for _ in range(batch_size):
            item_id = str(random.choice(self.item_ids))

            self.client.post(
                "/predict/rating/",
                json={"user_id": user_id, "item_id": item_id}
            )

        response_time = (time.time() - start_time) * 1000

        self.log_response_time(
            response_time,
            f"Batch Rating Predictions (n={batch_size})",
            {"batch_size": batch_size, "user_id": user_id}
        )

        @task(1)
        def test_healh_endpoint(self):
            start_time = time.time()
            self.client.get("/health/")
            response_time = (time.time() - start_time) * 1000

            self.log_response_time(
                response_time,
                "Health Check"
            )

@events.test_start.add_listener
def on_test_started(environment, **kwargs):
    print("Starting performance testing...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Performance testing completed.")

if __name__ == '__main__':
    pass