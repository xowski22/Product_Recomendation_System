import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import mlflow

from src.data.preprocessing import load_ml1m_data, preprocess_ratings, split_data
from src.data.dataset import RecommenderDataset
from src.models.model import MatrixFactorization
from grind_search import GrindSearch

def main():
    config_path = Path("../../config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ratings_df, movies_df = load_ml1m_data("../../data/raw/ml-1m")

    processed_ratings, user_mapping, item_mapping = preprocess_ratings(ratings_df)
    train_data, val_data = split_data(processed_ratings)

    train_dataset = RecommenderDataset(train_data)
    val_dataset = RecommenderDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    param_grind = {
        'num_users': [len(user_mapping)],
        "n_items": [len(item_mapping)],
        'embedding_dim': [50, 100, 150],
        'reg_lambda': [0.001, 0.01, 0.1]
    }

    grind_search = GrindSearch(
        model_class=MatrixFactorization,
        param_grind=param_grind,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )

    print("Starting grind search...")
    results = grind_search.fit()

    best_params = grind_search.get_best_params()
    print("\nBest parameters:")
    print(best_params)

    print("\nTraining final model with best parameters...")
    best_model = MatrixFactorization(**best_params)

    with open('best_model_config.yaml', 'w') as f:
        yaml.dump(best_params, f)

if __name__ == '__main__':
    main()