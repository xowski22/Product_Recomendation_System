import mlflow
import itertools
from typing import List, Dict, Any, Union
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

class GrindSearch:
    def __init__(
            self,
            model_class,
            param_grind: Dict[str, List[Any]],
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int,
            device: str = None
    ):
        self.model_class = model_class
        self.param_grind = param_grind
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []


    def _train_and_evaluate(
            self,
            model,
            params: Dict[str, Any],

) -> Dict[str, Union[float, Dict]]:
        model = model.to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params.get('learning_rate', 0.001)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.1
        )

        criterion = torch.nn.MSELoss()

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0.0
            for batch in self.train_loader:
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)

                predictions = model(user_ids, item_ids)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]

                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            avg_train_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    user_ids = batch['user_id'].to(self.device)
                    item_ids = batch['item_id'].to(self.device)
                    ratings = batch['rating'].to(self.device)

                    predictions = model(user_ids, item_ids)
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]

                    loss = criterion(predictions, ratings)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            mlflow.log_metrics({
                f"train_loss_epoch_{epoch}": avg_train_loss,
                f"val_loss_epoch_{epoch}": avg_val_loss
            }, step=epoch)

            return {
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'params': params
            }


    def fit(self) -> List[Dict[str, Any]]:
        param_combinations = ParameterGrid(self.param_grind)
        total_combinations = len(param_combinations)

        with mlflow.start_run(run_name="grind_search") as parent_run:
            for i, params in enumerate(param_combinations, 1):
                print(f"\nTrying combination {i}/{total_combinations}:")
                print(params)

                with mlflow.start_run(nested=True, run_name=f"combination_{i}") as child_run:
                    mlflow.log_params(params)

                    model = self.model_class(**params)

                    result = self._train_and_evaluate(model, params)
                    self.results.append(result)

                    mlflow.log_metric("best_val_loss", result["best_val_loss"])

                    self._plot_learning_curves(
                        result['train_losses'],
                        result['val_losses'],
                        params
                    )

        return self.results

    def _plot_learning_curves(
            self,
            train_losses: List[float],
            val_losses: List[float],
            params: Dict[str, Any]
    ) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Learning Curves\n{params}')
        plt.legend()
        plt.grid(True)

        plot_path = f"loss_plot_{params}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

    def get_best_params(self) -> Dict[str, Any]:
        best_result = min(self.results, key=lambda x: x["best_val_loss"])
        return best_result['params']