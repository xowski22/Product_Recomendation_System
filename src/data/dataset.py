import torch
from torch.utils.data import Dataset

class RecommenderDataset(Dataset):
    def __init__(self, ratings_df, transform=None):
        self.users = torch.tensor(ratings_df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(ratings_df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float)
        self.transform = transform

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        sample = {
            'user_id': self.users[idx],
            'item_id': self.items[idx],
            'rating': self.ratings[idx],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
