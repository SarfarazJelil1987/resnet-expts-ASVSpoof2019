import os
import torch
from pathlib import Path
from typing import Dict, Any

from config import Config
from data.dataloader import get_dataloader
from model.resnet import ResNet
from trainer.evaluator import Evaluator
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )                   
logger = logging.getLogger(__name__)

class TestManager:
    """Manages the testing process for the ASVspoof model."""
    
    def __init__(self, config_path: str):
        """
        Initialize the test manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def load_model(self) -> ResNet:
        """
        Load the trained model from checkpoint.
        
        Returns:
            ResNet: Loaded model
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If model loading fails
        """
        checkpoint_path = Path(self.config.checkpoint_path) / 'best_model.pth'
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model = ResNet(
                num_nodes=3,
                enc_dim=self.config.enc_dim,
                resnet_type=self.config.resnet_type,
                nclasses=self.config.num_classes
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def run_evaluation(self) -> Dict[str, float]:
        """
        Run the evaluation process.
        
        Returns:
            Dict containing evaluation metrics
        
        Raises:
            RuntimeError: If evaluation fails
        """
        try:
            # Initialize test loader
            test_loader = get_dataloader(self.config, 'dev')
            logger.info("Test dataloader initialized")

            # Load model
            model = self.load_model()

            # Initialize evaluator and run evaluation
            evaluator = Evaluator(self.config, model, test_loader)
            results = evaluator.evaluate()
            
            self._log_results(results)
            return results
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")

    def _log_results(self, results: Dict[str, float]) -> None:
        """
        Log evaluation results.
        
        Args:
            results: Dictionary containing evaluation metrics
        """
        logger.info("=== Evaluation Results ===")
        logger.info(f"Test EER: {results['eer']:.4f}")
        logger.info(f"Threshold: {results['threshold']:.4f}")
        if 'accuracy' in results:
            logger.info(f"Accuracy: {results['accuracy']:.4f}")

def main():
    """Main function to run the evaluation."""
    try:
        test_manager = TestManager('configs/config.yaml')
        results = test_manager.run_evaluation()
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
