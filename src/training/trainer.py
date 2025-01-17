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
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        user_ids = batch['user_id'].to(device)
        item_ids = batch['item_id'].to(device)
        ratings = batch['rating'].to(device)

        predictions = model(user_ids, item_ids, ratings)
        loss = criterion(predictions, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return {'loss': total_loss / len(train_loader)}

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                config: Dict[str, Any]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    mlflow.start_run()
    mlflow.log_param(config)

    for epoch in range(config['num_epochs']):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        mlflow.log_metrics(train_metrics, step=epoch)

        val_metrics = validate(model, val_loader, criterion, device)
        mlflow.log_metrics(val_metrics, step=epoch)

        print(f"Epoch {epoch + 1}/{config['num_epochs']} ")
        print(f"Train Loss: {train_metrics['loss']}")
        print(f"Val Loss: {val_metrics['loss']}")

        mlflow.end_run()
        return model
