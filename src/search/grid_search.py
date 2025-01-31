import mlflow
from typing import List, Dict, Any, Union
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import os
from pathlib import Path
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

class GridSearch:
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

        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.experiment_dir = Path(f'../search/experiments/{self.timestamp}')
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def _create_combination_dir(self, combination_number, params):
        """Creates directory for specifc parameter combination"""
        combination_dir = self.experiment_dir / f'combination_{combination_number}'
        combination_dir.mkdir(parents=True, exist_ok=True)

        with open(combination_dir / 'parameters.txt', 'w') as f:
            for key, value in params.items():
                f.write(f'{key}={value}\n')

        return combination_dir

    def _train_and_evaluate(
            self,
            model,
            params: Dict[str, Any],
            save_dir
) -> Dict[str, Union[float, Dict]]:
        """Trains model and saves results"""
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
            print(f"\nStarting epoch {epoch + 1}/{self.num_epochs}")
            model.train()
            epoch_train_loss = 0.0
            num_train_samples = 0

            for batch in self.train_loader:
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)

                optimizer.zero_grad()
                output = model(user_ids, item_ids)
                if isinstance(output, tuple):
                    predictions = output[0]
                else:
                    predictions = output

                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                num_train_samples += 1

            avg_train_loss = epoch_train_loss / num_train_samples if num_train_samples > 0 else 0
            train_losses.append(avg_train_loss)

            print(f"Epoch {epoch + 1} training completed - Avg train loss: {avg_train_loss:.4f}")

            model.eval()
            epoch_val_loss = 0.0
            num_val_samples = 0

            with torch.no_grad():
                for batch in self.val_loader:
                    user_ids = batch['user_id'].to(self.device)
                    item_ids = batch['item_id'].to(self.device)
                    ratings = batch['rating'].to(self.device)

                    predictions = model(user_ids, item_ids)
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]

                    loss = criterion(predictions, ratings)
                    epoch_val_loss += loss.item()
                    num_val_samples += 1

            avg_val_loss = epoch_val_loss / num_val_samples if num_val_samples > 0 else 0
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch + 1} validation completed - Avg val loss: {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_dir / 'best_model.pt')
                print(f"New best validation loss: {best_val_loss:.4f}")

            mlflow.log_metrics({
                f"train_loss_epoch_{epoch}": avg_train_loss,
                f"val_loss_epoch_{epoch}": avg_val_loss
            }, step=epoch)

            print(f"Completed epoch {epoch + 1}/{self.num_epochs}")

        return {
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'params': params
        }


    def fit(self) -> List[Dict[str, Any]]:
        """Performs grid search over all parameters"""
        param_combinations = ParameterGrid(self.param_grind)
        total_combinations = len(param_combinations)

        with open(self.experiment_dir / 'experiment_info.txt', 'w') as f:
            f.write(f'Experiment timestamp: {self.timestamp}\n')
            f.write(f'Total combinations: {total_combinations}\n')
            f.write('\nParameter grid:\n')
            for param, values in self.param_grind.items():
                f.write(f'{param}: {values}\n')

        with mlflow.start_run(run_name="grind_search") as parent_run:
            for i, params in enumerate(param_combinations, 1):
                print(f"\nTrying combination {i}/{total_combinations}:")
                print(params)

                combination_dir = self._create_combination_dir(i, params)

                with mlflow.start_run(nested=True, run_name=f"combination_{i}") as child_run:
                    mlflow.log_params(params)

                    model = self.model_class(**params)

                    result = self._train_and_evaluate(model, params, combination_dir)
                    self.results.append(result)

                    mlflow.log_metric("best_val_loss", result['best_val_loss'])

                    self._plot_learning_curves(
                        result['train_losses'],
                        result['val_losses'],
                        params,
                        combination_dir / 'final_learning_curves.png'
                    )

        self._save_results_summary()
        return self.results

    def _plot_learning_curves(
            self,
            train_losses: List[float],
            val_losses: List[float],
            params: Dict[str, Any],
            save_path: Path
    ) -> None:
        """Plots and saves learning curves for parameter combinations"""

        plt.figure(figsize=(10, 6))

        if len(train_losses) > 1:
            epochs = list(range(1, len(train_losses) +1))

            plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            plt.plot(epochs, train_losses, 'bo', markersize=4)
            plt.plot(epochs, val_losses, 'ro', markersize=4)

            plt.xticks(epochs)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.title(f'Learning Curves\n{params}')
            plt.legend()
            plt.grid(True)

            ymin = min(min(train_losses), min(val_losses))
            ymax = max(max(train_losses), max(val_losses))
            margin = (ymax - ymin) * 0.1
            plt.ylim(ymin - margin, ymax + margin)
        else:
            print("nothing to plot")

        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        mlflow.log_artifact(str(save_path))
        plt.close()

    def _save_results_summary(self):

        summary_path = self.experiment_dir / 'results_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f'Grid Search Results Summary\n')
            f.write(f'Timestamp: {self.timestamp}\n\n')

            sorted_results = sorted(self.results, key=lambda x: x['best_val_loss'])

            for i, result in enumerate(sorted_results, 1):
                f.write(f'\nRank {i}:\n')
                f.write(f'Parameters: {result["params"]}\n')
                f.write(f'Best validation loss: {result["best_val_loss"]:.6f}\n')
                f.write('-' * 50 + '\n')

    def get_best_params(self) -> Dict[str, Any]:
        best_result = min(self.results, key=lambda x: x["best_val_loss"])
        return best_result['params']