import pandas as pd
import numpy as np
from pathlib import Path
import pickle

"""Older """

def load_ml1m_data(data_dir: str):
    ratings = pd.read_csv(Path(data_dir) / 'ratings.dat',
                          sep='::',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'],
                          encoding='latin-1',
                          engine='python'
                          )

    movies = pd.read_csv(Path(data_dir) / 'movies.dat',
                         sep='::',
                         names=['movie_id', 'title', 'genres'],
                         encoding='latin-1',
                         engine='python'
                         )
    return ratings, movies

def preprocess_ratings(ratings_df: pd.DataFrame, save_mappings: bool = True, mappings_dir: str = "../data/mappings"):

    user_mapping = {id:idx for idx, id in enumerate(ratings_df['user_id'].unique())}
    movies_mapping = {id: idx for idx, id in enumerate(ratings_df['movie_id'].unique())}

    ratings_df['user_id'] = ratings_df['user_id'].map(user_mapping)
    ratings_df['movie_id'] = ratings_df['movie_id'].map(movies_mapping)

    ratings_df['normalized_rating'] = ratings_df['rating'] / 5.0

    ratings_df = ratings_df.rename(columns={'movie_id': 'item_id'})

    if save_mappings:
        mappings_path = Path(mappings_dir)
        mappings_path.mkdir(parents=True, exist_ok=True)

        with open(mappings_path / 'user_mapping.pkl', 'wb') as f:
            pickle.dump(user_mapping, f)

        with open(mappings_path / 'item_mapping.pkl', 'wb') as f:
            pickle.dump(movies_mapping, f)

    return ratings_df, user_mapping, movies_mapping

def split_data(ratings_df: pd.DataFrame, val_ratio: float = 0.2):

    train_data = ratings_df.sample(frac=1-val_ratio, random_state=42)
    test_data = ratings_df.drop(train_data.index)

    return train_data, test_data