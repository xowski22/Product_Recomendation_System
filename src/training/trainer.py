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
        user_ids = batch['user_id'].to(device)
        item_ids = batch['item_id'].to(device)
        ratings = batch['rating'].to(device)

        #redicting ratings by foward pass
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

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    mlflow.start_run()

    for key, value in config.items():
        mlflow.log_param(key, value)

    for epoch in range(config['num_epochs']):
        #trenujemy przez jedną epokę
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        #zapis metryk z treningu
        mlflow.log_metrics(train_metrics, step=epoch)

        #walidacja modelu
        val_metrics = validate(model, val_loader, criterion, device)
        mlflow.log_metrics(val_metrics, step=epoch)

        print(f"Epoch {epoch + 1}/{config['num_epochs']} ")
        print(f"Train Loss: {train_metrics['loss']}")
        print(f"Val Loss: {val_metrics['loss']}")

        mlflow.end_run()
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
            loss = criterion(predictions, ratings)
            total_loss += loss.item()

    return {'loss': total_loss / len(val_loader)}