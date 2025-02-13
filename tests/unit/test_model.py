import pytest
import torch
import torch.nn as nn
from src.models.model import MatrixFactorization

"""Need to change it asap, wont work with current model"""

@pytest.fixture
def sample_model():
    return MatrixFactorization(
        num_users=10,
        n_items=20,
        embedding_dim=5,
        reg_lambda=0.01
    )

def test_model_init():
    model = MatrixFactorization(num_users=100, n_items=50, embedding_dim=10, reg_lambda=0.01)

    assert model.user_embeddings.weight.shape == (100, 10)
    assert model.item_embeddings.weight.shape == (50, 10)

    assert model.global_bias.weight.shape == (1,)
    assert model.user_bias.weight.shape == (100,)
    assert model.item_bias.weight.shape == (50,)

    assert isinstance(model.user_bn, nn.BatchNorm1d)
    assert isinstance(model.item_bn, nn.BatchNorm1d)
    assert model.user_bn.num_features == 10
    assert model.item_bn.num_features == 10

    assert isinstance(model.dropout, nn.Dropout)
    assert model.dropout == 0.2

def test_forward_pass_training(sample_model):
    users = torch.tensor([0, 1, 2], dtype=torch.long)
    items = torch.tensor([0, 1, 2], dtype=torch.long)

    sample_model.train()
    predictions, user_embeds, item_embeds = sample_model(users, items)

    assert predictions.shape == (3,)
    assert user_embeds.shape == (3,5)
    assert item_embeds.shape == (3,5)
    assert predictions.requires_grad
    assert user_embeds.requires_grad
    assert item_embeds.requires_grad

def test_forward_pass_eval(sample_model):
    users = torch.tensor([0, 1, 2], dtype=torch.long)
    items = torch.tensor([0, 1, 2], dtype=torch.long)

    sample_model.eval()
    with torch.no_grad():
        predictions = sample_model(users, items)

        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == (3,)
        assert predictions.dtype == torch.float32
        assert not predictions.requires_grad

def test_batch_normalization(sample_model):
    users = torch.tensor([0, 1, 2], dtype=torch.long)
    items = torch.tensor([0, 1, 2], dtype=torch.long)

    sample_model.train()
    _, user_embeds_train, _ = sample_model(users, items)

    sample_model.eval()
    with torch.no_grad():
        _, user_embeds_eval, _ = sample_model(users, items)

    assert not torch.allclose(user_embeds_train, user_embeds_eval)

