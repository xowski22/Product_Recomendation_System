import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, n_items: int, embedding_dim: int = 100):
        super.__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(self, users_ids: torch.Tensor, items_ids: torch.Tensor) -> torch.Tensor:
        user_embeds = self.user_embeddings(users_ids)
        item_embeds = self.item_embeddings(items_ids)
        return torch.sum(user_embeds * item_embeds, dim=1)