import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, n_items: int, embedding_dim: int = 100, reg_lambda: float = 0.1):
        super().__init__()
        #embedding layers init
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)

        #weights init
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

        #regularization parameter
        self.reg_lambda = reg_lambda

        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(n_items))



    def forward(self, users_ids: torch.Tensor, items_ids: torch.Tensor, user_bias: torch.Tensor, item_bias: torch.Tensor) -> torch.Tensor:
        #get embeddings from users and items
        user_embeds = self.user_embeddings(users_ids)
        item_embeds = self.item_embeddings(items_ids)

        #get bias for users and items
        user_bias = self.user_bias[users_ids]
        item_bias = self.item_bias[items_ids]

        prediction = (
                self.global_bias +
                user_bias +
                item_bias +
                torch.sum(user_embeds * item_embeds, dim=1)
                )

        if self.training:
            return prediction, user_embeds, item_embeds
        return prediction