import pickle
from typing import Dict, Any, Tuple
from pathlib import Path
from torch.utils.data import DataLoader
from ..dataset import RecommenderDataset
from .base_loader import BaseDataLoader
from .movielens_loader import MovieLensLoader
from ...api.app import user_mapping


class DataLoaderFactory:
    @staticmethod
    def get_loader(config: Dict[str, Any]) -> BaseDataLoader:
        data_type = config['data']['type'].lower()

        if data_type == 'movielens':
            return MovieLensLoader(config)
        else:
            raise ValueError(f"Unsuported data type: {data_type}")


def save_mappings(user_mapping: Dict[int, int], item_mapping: Dict[int, int], config: Dict[str, Any]):
    mappings_dir = Path(config['data']['mappings_dir'])
    mappings_dir.mkdir(parents=True, exist_ok=True)

    with open(mappings_dir / 'user_mapping.plk', 'wb') as f:
        pickle.dump(user_mapping, f)

    with open(mappings_dir / 'item_mapping.plk', 'wb') as f:
        pickle.dump(item_mapping, f)


def create_train_val_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dict, Dict]:
    loader = DataLoaderFactory.get_loader(config)

    ratings_df, _ = loader.load_data()
    processed_df, user_mapping, item_mapping = loader.preprocess_data(ratings_df)
    train_data, val_data = loader.split_data(processed_df)

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

    return train_loader, val_loader, user_mapping, item_mapping