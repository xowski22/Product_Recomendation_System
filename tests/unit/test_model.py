import pytest
import torch
import torch.nn as nn
from src.models.model import MatrixFactorization

"""test for older model config, before chages"""

@pytest.fixture
def sample_model():
    return MatrixFactorization(
        num_users=10,
        n_items=20,
        embedding_dim=5,
        reg_lambda=0.01
    )

def test_model_init():
    model = MatrixFactorization(num_users=100, n_items=50, embedding_dim=10)

    assert model.user_embeddings.weight.shape == (100, 10)
    assert model.item_embeddings.weight.shape == (50, 10)

    assert model.global_bias.weight.shape == (1,)
    assert model.user_bias.weight.shape == (100,)
    assert model.item_bias.weight.shape == (50,)

def test_forward_pass(sample_model):
    users = torch.tensor([0, 1, 2], dtype=torch.long)
    items = torch.tensor([0, 1, 2], dtype=torch.long)

    sample_model.eval()
    with torch.no_grad():
        predictions = sample_model(users, items)

        assert predictions.shape == (3,)
        assert predictions.dtype == torch.float32

    sample_model.train()
    predictions, user_embeds, item_embeds = sample_model(users, items)

    assert predictions.shape == (3,)
    assert user_embeds.shape == (3,5)
    assert item_embeds.shape == (3,5)