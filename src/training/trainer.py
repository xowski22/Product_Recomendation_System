import torch
import torch.nn as nn
from sympy.matrices.expressions.kronecker import validate
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import mlflow
from typing import Dict, Any

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device:str) -> Dict[str, float]:
    model.train() #switching model into train mode
    total_loss = 0.0
    for batch in train_loader:
        user_ids = batch['user_id'].to(device, non_blocking=True)
        item_ids = batch['item_id'].to(device, non_blocking=True)
        ratings = batch['rating'].to(device, non_blocking=True)

        #predicting ratings by foward pass
        predictions, user_embeds, item_embeds = model(user_ids, item_ids)
        #calculating mse loss
        mse_loss = criterion(predictions, ratings)

        reg_loss = model.reg_lambda + (torch.norm(user_embeds)**2 + torch.norm(item_embeds)**2)

        loss = mse_loss + reg_loss

        #backward pass - updating weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return {'loss': total_loss / len(train_loader)}#średnia strata z jednej epoki

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                config: Dict[str, Any]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    learning_rate = config.get('learning_rate', 0.001)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    'min',
                                                           patience=3,
                                                           factor=0.2,
                                                           min_lr=1e-6,
                                                           )
    criterion = nn.MSELoss()

    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                mlflow.log_param(f"{key}.{sub_key}", sub_value)
        else:
            mlflow.log_param(key, value)

    num_epochs = config.get('num_epochs', 15)

    for epoch in range(num_epochs):

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        val_metrics = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs} ")
        print(f"Train Loss: {train_metrics['loss']}")
        print(f"Val Loss: {val_metrics['loss']}")
        print("-" * 40)

        mlflow.log_metrics({
            f"train_loss_{epoch}": train_metrics['loss'],
            f"val_loss_{epoch}": val_metrics['loss']
        }, step=epoch)

        scheduler.step(val_metrics['loss'])

    return model

def validate(model: nn.Module,
             val_loader: DataLoader,
             criterion: nn.Module,
             device:str) -> float:
    model.eval()
    total_loss = 0.0
    with (torch.no_grad()):
        for batch in val_loader:
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            ratings = batch['rating'].to(device)

            predictions = model(user_ids, item_ids)
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            loss = criterion(predictions, ratings)
            total_loss += loss.item()

    return {'loss': total_loss / len(val_loader)}