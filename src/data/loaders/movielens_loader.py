import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from .base_loader import BaseDataLoader


class MovieLensLoader(BaseDataLoader):
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data_path = Path(self.config['data']['path'])

        ratings = pd.read_csv(
            data_path / self.config['data']['ratings_file'],
            sep= self.config['data']['separator'],
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1',
            engine='python'
            )

        movies = pd.read_csv(
            data_path / self.config['data']['movies_file'],
            sep= self.config['data']['separator'],
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1',
            engine='python'
        )

        return ratings, movies

    def preprocess_data(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int ,int], Dict[int ,int]]:

        if self.config['data']['min_user_interactions'] > 0:
            user_counts = ratings_df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.config['data']['min_user_interactions']].index
            ratings_df = ratings_df[ratings_df['user_id'].isin(valid_users)]

        user_mapping = {id: idx for idx, id in enumerate(ratings_df['user_id'].unique())}
        item_mapping = {id: idx for idx, id in enumerate(ratings_df['movie_id'].unique())}

        ratings_df['user_id'] = ratings_df['user_id'].map(user_mapping)
        ratings_df['movie_id'] = ratings_df['movie_id'].map(item_mapping)

        if self.config['preprocessing']['remove_duplicates']:
            ratings_df = ratings_df.drop_duplicates(['user_id', 'movie_id'])

        ratings_df = ratings_df.rename(columns={'movie_id': 'item_id'})

        return ratings_df, user_mapping, item_mapping

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        val_ratio = self.config['data']['validation_size']
        train_data = data.sample(frac=1-val_ratio, random_state=42)
        val_data = data.drop(train_data.index)

        return train_data, val_data