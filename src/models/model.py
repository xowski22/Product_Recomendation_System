import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, n_items: int, embedding_dim: int, reg_lambda: float):

        """
        Neural matrix factorization model with the following features:

        -Embedding layers for users and items
        -Batch normalization for training stability
        -Dropout for regularization
        -Global and user/item specific bias terms
        """

        super().__init__()
        #embedding layers init
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        self.dropout = nn.Dropout(0.2)

        self.user_bn = nn.BatchNorm1d(embedding_dim)
        self.item_bn = nn.BatchNorm1d(embedding_dim)

        #weights init
        nn.init.normal_(self.user_embeddings.weight, std=0.001)
        nn.init.normal_(self.item_embeddings.weight, std=0.001)

        #regularization parameter
        self.reg_lambda = reg_lambda

        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(n_items))

        nn.init.normal_(self.user_bias, std=0.001)
        nn.init.normal_(self.item_bias, std=0.001)

    def forward(self, users_ids: torch.Tensor, items_ids: torch.Tensor) -> torch.Tensor:
        #get embeddings from users and items
        user_embeds = self.user_bn(self.dropout(self.user_embeddings(users_ids)))
        item_embeds = self.item_bn(self.dropout(self.item_embeddings(items_ids)))

        #get bias for users and items
        user_bias = self.user_bias[users_ids]
        item_bias = self.item_bias[items_ids]

        # prediction = (
        #         self.global_bias +
        #         user_bias +
        #         item_bias +
        #         torch.sum(user_embeds * item_embeds, dim=1)
        #         )

        prediction = torch.sum(user_embeds * item_embeds, dim=1)
        prediction.add_(self.global_bias.squeeze())
        prediction.add_(self.user_bias)
        prediction.add_(self.item_bias)

        if self.training:
            return prediction, user_embeds, item_embeds
        return prediction