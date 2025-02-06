import yaml
import torch
from torch.utils.data import DataLoader
import mlflow
from pathlib import Path

from data.preprocessing import load_ml1m_data, preprocess_ratings, split_data
from data.dataset import RecommenderDataset
from src.api.app import user_mapping
from src.data.loaders.loader_factory import create_train_val_dataloaders
from src.models.model import MatrixFactorization
from src.training.trainer import train_model


def main():

    ROOT_DIR = Path(__file__).parent.parent

    (ROOT_DIR / 'data' / 'mappings').mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / 'models' / 'checkpoints').mkdir(parents=True, exist_ok=True)

    config_path = ROOT_DIR / 'config' / "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['data']['path'] = str(ROOT_DIR / 'data' / 'raw' / 'ml-1m')
    config['data']['mappings_path'] = str(ROOT_DIR / 'data' / 'mappings')

    train_loader, val_loader, user_mapping, item_mapping = create_train_val_dataloaders(config)

    model = MatrixFactorization(
        num_users=len(user_mapping),
        n_items=len(item_mapping),
        embedding_dim=config['model']['embedding_dim'],
        reg_lambda=config['model']['reg_lambda']
    )

    trained_model = train_model(model, train_loader, val_loader, config['training'])

    model_save_path = ROOT_DIR / 'models' / 'checkpoints' / 'best_model.pt'
    torch.save(trained_model.state_dict(), model_save_path)

if __name__ == '__main__':
    main()