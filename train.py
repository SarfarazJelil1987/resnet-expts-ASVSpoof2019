import torch
import os
import random
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
from torch.utils.data import DataLoader

from config import Config
from data.dataloader import get_dataloader
from model.resnet import ResNet
from trainer.trainer import Trainer
from utils.logger import Logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

torch.set_default_tensor_type(torch.FloatTensor)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
@dataclass
class TrainingConfig:
    """Configuration for training setup."""
    config_path: str
    experiment_name: str
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    resume_checkpoint: Optional[str] = None

class TrainingPipeline:
    """Manages the complete training pipeline."""

    def __init__(self, training_config: TrainingConfig):
        """
        Initialize training pipeline.

        Args:
            training_config: Training configuration
        """
        self.training_config = training_config
        self.config = self._load_config()
        self.device = torch.device(training_config.device)

        # Set random seeds
        self._set_random_seeds()

        # Initialize components
        self.train_loader, self.val_loader = self._setup_dataloaders()
        self.model = self._setup_model()
        self.logger = self._setup_logger()
        self.trainer = self._setup_trainer()
        

    def _load_config(self) -> Config:
        """Load and validate configuration."""
        try:
            config = Config(self.training_config.config_path)
            logger.info(f"Loaded configuration from {self.training_config.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.training_config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set random seed to {seed}")

    def _setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Initialize train and validation dataloaders."""
        try:
            train_loader = get_dataloader(self.config, 'train')
            val_loader = get_dataloader(self.config, 'dev')

            logger.info(f"Initialized dataloaders - "
                       f"Train: {len(train_loader)} batches, "
                       f"Val: {len(val_loader)} batches")

            return train_loader, val_loader
        except Exception as e:
            logger.error(f"Failed to setup dataloaders: {e}")
            raise

    def _setup_model(self) -> ResNet:
        """Initialize model architecture."""
        try:
            model = ResNet(
                num_nodes=3,
                enc_dim=self.config.enc_dim,
                resnet_type=self.config.resnet_type,
                nclasses=self.config.num_classes
            )

            # Move model to device
            model = model.to(self.device)

            # Load checkpoint if specified
            if self.training_config.resume_checkpoint:
                self._load_checkpoint(model)

            logger.info(f"Initialized {self.config.resnet_type} model")
            return model

        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise

    def _load_checkpoint(self, model: ResNet) -> None:
        """Load model checkpoint."""
        try:
            checkpoint_path = Path(self.training_config.resume_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found at {checkpoint_path}"
                )

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _setup_logger(self) -> Logger:
        """Initialize training logger."""
        try:
            #log_dir = Path(self.config.log_dir) / self.training_config.experiment_name
            logger = Logger(self.config)
            #logger.info(f"Initialized logger at {logger.config.log_dir}")
            return logger
        except Exception as e:
            logger.error(f"Failed to setup logger: {e}")
            raise

    def _setup_trainer(self) -> Trainer:
        """Initialize trainer."""
        try:
            trainer = Trainer(
                config=self.config,
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                logger=self.logger
            )
            logger.info("Initialized trainer")
            return trainer
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise

    def train(self) -> None:
        """Execute training pipeline."""
        try:
            logger.info("Starting training...")
            self.trainer.train()
            logger.info("Training completed successfully")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._handle_interrupt()

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        finally:
            self._cleanup()

    def _handle_interrupt(self) -> None:
        """Handle training interruption."""
        try:
            # Save checkpoint
            self.trainer.save_checkpoint(is_best=False, filename='interrupt.pth')
            logger.info("Saved interrupt checkpoint")
        except Exception as e:
            logger.error(f"Failed to save interrupt checkpoint: {e}")

    def _cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Close logger
            if hasattr(self.logger, 'close'):
                self.logger.close()

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

def main():
    """Main entry point for training."""
    try:
        # Setup training configuration
        training_config = TrainingConfig(
            config_path='configs/config.yaml',
            experiment_name='2D_ILRCC_resnet_expt',
            resume_checkpoint=None
        )

        # Initialize and run training pipeline
        pipeline = TrainingPipeline(training_config)
        pipeline.train()

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == '__main__':
    main()
