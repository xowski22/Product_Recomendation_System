from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict, Any

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
