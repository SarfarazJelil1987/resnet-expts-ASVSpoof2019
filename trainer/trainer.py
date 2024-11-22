from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import compute_eer
from utils import logger

@dataclass
class TrainingConfig:
    device: str
    lr: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    num_epochs: int
    lr_decay: float
    lr_decay_interval: int
    checkpoint_path: str

class Trainer:
    def __init__(self, config: TrainingConfig, model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 logger: logger):
        """Initialize the trainer with model, data loaders, and configuration.

        Args:
            config: Training configuration
            model: Neural network model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            logger: Logger instance for tracking metrics
        """
        self.config = config
        self.model = self._setup_model(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._setup_optimizer()
        self.best_eer = float('inf')
        self.checkpoint_path = Path(config.checkpoint_path)

    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for training."""
        return model.to(self.config.device)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Initialize the optimizer with configured parameters."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay
        )

    def _process_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of data for model input.

        Args:
            batch: Tuple of (features, _, labels)

        Returns:
            Processed features and labels
        """
        features, _, labels = batch
        features = features.unsqueeze(1).float().to(self.config.device)
        labels = labels.to(self.config.device)
        return features, labels

    def _train_step(self, features: torch.Tensor,
                    labels: torch.Tensor) -> float:
        """Perform a single training step.

        Args:
            features: Input features
            labels: Target labels

        Returns:
            Loss value for the step
        """
        self.optimizer.zero_grad()
        _, outputs = self.model(features)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, epoch: int) -> float:
        """Train the model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0

        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch in pbar:
                features, labels = self._process_batch(batch)
                loss = self._train_step(features, labels)
                total_loss += loss

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})

        return total_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, float]:
        """Validate the model on validation set.

        Returns:
            Tuple of (validation loss, EER)
        """
        self.model.eval()
        total_loss = 0
        scores = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                features, batch_labels = self._process_batch(batch)
                _, outputs = self.model(features)
                loss = self.criterion(outputs, batch_labels)

                scores.extend(outputs[:, 1].cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
                total_loss += loss.item()

        val_loss = total_loss / len(self.val_loader)
        eer, _ = compute_eer(scores, labels)

        return val_loss, eer

    def _adjust_learning_rate(self, epoch: int) -> None:
        """Adjust learning rate based on epoch."""
        lr = self.config.lr * (self.config.lr_decay **
                              (epoch // self.config.lr_decay_interval))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _save_checkpoint(self, epoch: int, eer: float) -> None:
        """Save model checkpoint."""
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'eer': eer,
        }
        torch.save(checkpoint, self.checkpoint_path / 'best_model.pth')

    def train(self) -> None:
        """Train the model for the specified number of epochs."""
        for epoch in range(self.config.num_epochs):
            try:
                # Adjust learning rate
                self._adjust_learning_rate(epoch)

                # Train and validate
                train_loss = self.train_epoch(epoch)
                val_loss, eer = self.validate()

                # Log metrics
                self.logger.log_training(epoch, train_loss, val_loss, eer)

                # Save best model
                if eer < self.best_eer:
                    self.best_eer = eer
                    self._save_checkpoint(epoch, eer)

            except Exception as e:
                print(f"Error during training: {e}")
                raise
