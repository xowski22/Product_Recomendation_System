Product Recommendation System

Overview

Project implements a product recommendation system using collaborative filtering with matrix factorization. The system is built with PyTorch and provides both a training pipeline and a RestAPI for making predictions. Collaborative filtering provides personalized product recommendations based on user interactions and ratings.

Project Goals
- 
- Implement a recommendation system using matrix factorization
- Provide accurate product recommendations and rating predictions
- Create a production-ready API for real-time recommendations
- Support both single predictions and batch recommendations
- Ensure robust model evaluation and experimentation capabilities
- Enable easy integration with e-commerce platforms

Project Structure
-

product-recommendation/
├── config/
│   └── config.yaml         # Configuration parameters
├── src/
│   ├── api/               # FastAPI implementation 
│   ├── data/              # Data processing modules
│   ├── models/            # Model architecture
│   └── training/          # Training pipeline
├── tests/                 # Unit and integration tests
├── notebooks/            # Analysis notebooks
└── experiments/          # Experiment tracking

Instalation
-

Prerequisites
-

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU support)

Setup
-

1. Clone the repository: 
    git clone https://github.com/xowski22/Product_Recomendation_System.git
    cd Product_Recommendation_System
2. Create and activate virtual environment:
   python -m venv .venv
   source .venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt

Usage
-

Training the Model
-

1. Configure parameters in config.yaml
2. Run training:
    python /src/train.py

Running the API Server
-

1. Start the FastAPI server:
    python run_api.py
2. Access the API documentation http://localhost:8080/docs

API Examples
-

Get Rating Prediction
-

import requests

response = response.post(
    "http://localhost:8080/predict/rating/",
    json={
    "user_id": "1",
    "item_id": "123"
    }
)

print(response.json())

Get Product Recomendations
-

response = requests.port(
    "http://localhost:8080/recommend/",
    json={
    "user_id": "1",
    "n_recommendations": 5
    }
)

Model Architecture
-

The system uses a Matrix Factorization model which has the following features:

- Embedding layers for user and items
- Batch normalization for better training stability
- Dropout for regularization
- Global and user/item specific bias terms

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, n_items, embedding_dim=100):
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(n_items))


Experimental Results
-

Hyperparameter Optimization
-

Best performing configuration:

- Embedding dimention: 50
- Regularization lambda: 0.001
- Learing rate: 0.001
- Batch size: 64

Model Perfmance:

- Training loss: 0.8077
- Validation loss: 0.8268
- Average prediction time: 25ms
- Memory usage: ~500MB

Load Testing Results
-

Performance metrics under load:

- Average response time: 25ms
- Throughput: 200 requests/second
- 99th percentile latency: 75ms
- Concurrent users supported: 1000+

Development
-

Running Tests
-

pytest tests/

Load Testing
-

locust -f locustfile.py

Future improvements
-

1. Implement hybrid recommendation approach combining collaborative and content-based filtering
2. Add support for cold-start problems with new products/users
3. Implement A/B testing capabilities
4. Add real-time model updating based on user interactions
5. Implement caching for frequently requested predictions
6. Add support for seasonal and temporal patterns in recommendations
7. Implement personalized recommendation explanations

Contributing
-

Contributions are welcome! Feel free to submit a Pull Request. For major changes, please open an issue first to discuss what would you like to change.
