Product Recommendation System
-

Overview
-

Project implements a product recommendation system using collaborative filtering with matrix factorization. The system is built with PyTorch and provides both a training pipeline and a RestAPI for making predictions. The architecture supports flexible data loading with extensible model training and provides personalized product recommendations based on user interactions and ratings.

Key features
- 
- Modular data loading architecture supporting multable data sources
- Matrix factorization-based collaborative filtering
- Robust training pipeline with hyperparameter optimization
- Production-ready RestAPI with FastAPI
- Comprehensive testing suite and performance monitoring
- Load testing capabilities with Locust

Project Structure
-

product-recommendation/ \
├── config/ \
│   └── config.yaml              # Configuration parameters \
├── src/ \
│   ├── api/                     # FastAPI implementation \
│   ├── data/ \
│   │   ├── loaders/            # Data loading modules \
│   │   │   ├── base_loader.py  # Abstract base loader \
│   │   │   ├── movielens_loader.py  # MovieLens implementation \
│   │   │   └── loader_factory.py    # Factory for loader creation \
│   │   ├── preprocessing.py    # Older datapreprocessing utilities \
│   │   └── dataset.py         # PyTorch dataset implementations \
│   ├── models/                 # Model architectures \
│   └── training/              # Training pipeline \
├── tests/                     # Unit and integration tests \
├── notebooks/                # Analysis notebooks \
└── experiments/             # Experiment tracking 

Data Loading Architecture
-

The system implements a flexible data loading architecture:

```

class BaseDataLoader(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """returns tuple containing (ratings_df, items_df"""
        pass

    @abstractmethod
    def preprocess_data(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
        """
        args
         ratings_df : raw ratings dataframe

        :returns
            tuple containing
                processed ratings dataframe
                user id mapping dictionary
                item id mapping dictionary
        """
        pass

    @abstractmethod
    def split_data(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        args
            data: preprocessed dataframe to split
        returns
            tuple containing train_df, val_df
        """
        pass
   ```

New data sources can be added by implementing the BaseDataLoader interface:

```angular2html
class MovieLensLoader(BaseDataLoader):
    def load_data(self):
        # Implementation for MovieLens dataset
        pass
```

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
    ```
    git clone https://github.com/xowski22/Product_Recomendation_System.git
    cd Product_Recommendation_System
   ```
2. Create and activate virtual environment:
   ```
    python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
    ```
   pip install -r requirements.txt
    ```
Usage
-

Training the Model
-

1. Configure dataset and parameters in config.yaml
    ```
   data:
        typetype: "movielens" 
        path: "data/raw/ml-1m"
        min_user_interactions: 5
        validation_size: 0.1
   ```
2. Run training:
    ```
    python /src/train.py
    ```
Running the API Server
-

1. Start the FastAPI server:
    ```
    python run_api.py
   ```
2. Access the API documentation http://localhost:8080/docs

API Examples
-

Get Rating Prediction
-
```
response = response.post(
    "http://localhost:8080/predict/rating/",
    json={
    "user_id": "1",
    "item_id": "123"
    }
)

print(response.json())
```
Get Product Recommendations
-
```
response = requests.port(
    "http://localhost:8080/recommend/",
    json={
    "user_id": "1",
    "n_recommendations": 5
    }
)
```
Model Architecture
-

The system uses a Matrix Factorization model which has the following features:

- Embedding layers for user and items
- Batch normalization for better training stability
- Dropout for regularization
- Global and user/item specific bias terms


Experimental Results
-

Hyperparameter Optimization
-

Best performing configuration:

- Embedding dimension: 50
- Regularization lambda: 0.001
- Learning rate: 0.001
- Batch size: 64

Model Performance:

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
```
pytest tests/
```
Load Testing
-
```
locust -f locustfile.py
```
Future improvements
-

1. Additional data loader implementations for different data sources
2. Enhanced preprocessing pipeline with configurable steps
3. Support for cold start scenarios
4. Real-time model updating
5. Response catching layer
6. A/B testing framework
7. Recommendation explanations

Contributing
-

Contributions are welcome! Feel free to submit a Pull Request. For major changes, please open an issue first to discuss what would you like to change.

License
-

MIT
