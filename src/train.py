import yaml
import torch
from torch.utils.data import DataLoader
import mlflow
from pathlib import Path

from data.preprocessing import load_ml1m_data, preprocess_ratings, split_data
from data.dataset import RecommenderDataset
from src.models.model import MatrixFactorization
from src.training.trainer import train_model


def main():

    ROOT_DIR = Path(__file__).parent.parent

    (ROOT_DIR / 'data' / 'mappings').mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / 'models' / 'checkpoints').mkdir(parents=True, exist_ok=True)

    config_path = ROOT_DIR / 'config' / "config.yaml"
    data_path = ROOT_DIR / 'data' / 'raw' / 'ml-1m'
    mappings_dir = ROOT_DIR / 'data' / 'mappings'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ratings_df, _ = load_ml1m_data(data_path)
    processed_df, user_mapping, item_mapping = preprocess_ratings(ratings_df)
    train_data, val_data = split_data(processed_df)

    train_dataset = RecommenderDataset(train_data)
    val_dataset = RecommenderDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True)

    model = MatrixFactorization(num_users=len(user_mapping),
                                n_items=len(item_mapping),
                                embedding_dim=config['model']['embedding_dim'],
                                reg_lambda=config['model']['reg_lambda']
                                )

    trained_model = train_model(model, train_loader, val_loader, config['training'])

    model_save_path = ROOT_DIR / 'models' / 'checkpoints' / 'best_model.pt'
    torch.save(trained_model.state_dict(), model_save_path)

if __name__ == '__main__':
    main()