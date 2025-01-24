import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from src.data.preprocessing import load_ml1m_data, preprocess_ratings, split_data

@pytest.fixture
def sample_ratings_data():
    return pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3],
        'movie_id': [101, 102, 101, 103, 102],
        'rating': [5.0, 3.0, 4.0, 2.0, 5.0],
        'timestamp': [100000000, 100000001, 100000002, 100000003, 10000004]
    })

@pytest.fixture
def sample_movies_data():
    return pd.DataFrame({
        'movie_id': [101, 102, 103],
        'title': ['Movie1', 'Movie2', 'Movie3'],
        'genres': ['Action', 'Comedy', 'Drama']
    })

def test_load_ml1m_data(tmp_path):
    ratings_data = pd.DataFrame({
        'user_id': [1, 2],
        'movie_id': [101, 102],
        'rating': [5.0, 3.0],
        'timestamp': [10000000, 100000001],
    })

    movies_data = pd.DataFrame({
        'movie_id': [101, 102],
        'title': ['Test Movie 1', 'Test Movie 2'],
        'genres': ['Action', 'Comedy']
    })

    ratings_data.to_csv(tmp_path / 'ratings.csv', sep=":", index=False, header=False)
    movies_data.to_csv(tmp_path / 'movies.csv',sep=":", index=False, header=False)

    loaded_ratings, loaded_movies = load_ml1m_data(str(tmp_path))

    assert loaded_ratings.shape[0] == 2
    assert loaded_movies.shape[0] == 2
    assert list(loaded_ratings.columns) == ['user_id', 'movie_id', 'rating', 'timestamp']
    assert list(loaded_movies.columns) == ['movie_id', 'title', 'genres']

def test_preprocess_ratings(sample_ratings_data):
    processed_df, user_mapping, movie_mapping = preprocess_ratings(sample_ratings_data)

    assert len(user_mapping) == len(sample_ratings_data['user_id'].unique())
    assert len(movie_mapping) == len(sample_ratings_data['movie_id'].unique())

    assert all(0 <= rating <= 1 for rating in processed_df['normalized_rating'])

    assert all(isinstance(id_, (int, np.integer)) for id_ in processed_df['user_id'])
    assert all(isinstance(id_, (int, np.integer)) for id_ in processed_df['movie_id'])

    assert processed_df['user_id'].max() < len(user_mapping)
    assert processed_df['movie_id'].max() < len(movie_mapping)

def test_split_data(sample_ratings_data):
    val_ratio = 0.2
    train_data, test_data = split_data(sample_ratings_data, val_ratio)

    expected_test_size = int(len(sample_ratings_data) * val_ratio)
    assert len(test_data) == expected_test_size
    assert len(train_data) == len(sample_ratings_data) - expected_test_size

    assert set(train_data.index).isdisjoint(set(test_data.index))
    assert len(train_data) + len(test_data) == len(sample_ratings_data)

"""
Rewrite edge cases functions
"""


# def test_preprocess_ratings_edge_cases(sample_ratings_data):
#     empty_df = pd.DataFrame(columns=sample_ratings_data.columns)
#     with pytest.raises(Exception):
#         preprocess_ratings(empty_df)
#
#     df_with_na = sample_ratings_data.copy()
#     df_with_na.loc[0, 'rating'] = None
#     with pytest.raises(Exception):
#         preprocess_ratings(df_with_na)

# def test_split_data_edge_cases(sample_ratings_data):
#     with pytest.raises(Exception):
#         split_data(sample_ratings_data, 1.5)
#
#     tiny_df = sample_ratings_data.head(1)
#     train_data, test_data = split_data(tiny_df, 0.2)
#     assert len(train_data) + len(test_data) == len(sample_ratings_data)