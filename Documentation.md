Project Recommendation System Documentation

Part 1: Tutorials

Getting Started with Recommendations

1.1 Basic Concepts

- What is collaborative filtering?
- How matrix factorization works
- Understanding user-item interactions
- Basic recommendation concepts

1.2 Your First Recommendation

"""
import requests

base_url = "http://localhost:8080"

response = requests.post(

    f"{base_url}/recommend",
    json={
        "user_id": "1",
        "n_recommendations": 5

    }

)
"""