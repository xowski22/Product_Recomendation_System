import pandas as pd
import numpy as np
from pathlib import Path

def load_ml1m_data(data_dir: str):
    ratings = pd.read_csv(Path(data_dir) / 'ratings.csv',
                          sep='::',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'],
                          engine='python'
                          )

    movies = pd.read_csv(Path(data_dir) / 'movies.csv',
                         sep='::',
                         names=['movie_id', 'title', 'genres'],
                         engine='python'
                         )
    return ratings, movies

def preprocess_ratings(ratings_df: pd.DataFrame):

    user_mapping = {id:idx for idx, id in enumerate(ratings_df['user_id'].unique())}
    movies_mapping = {id: idx for idx, id in enumerate(ratings_df['movie_id'].unique())}

    ratings_df['user_id'] = ratings_df['user_id'].map(user_mapping)
    ratings_df['movie_id'] = ratings_df['movie_id'].map(movies_mapping)

    ratings_df['normalized_rating'] = ratings_df['rating'] / 5.0

    return ratings_df, user_mapping, movies_mapping

def split_data(ratings_df: pd.DataFrame, val_ratio: float = 0.2):

    train_data = ratings_df.sample(frac=1-val_ratio, random_state=42)
    test_data = ratings_df.drop(train_data.index)

    return train_data, test_data