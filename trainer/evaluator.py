import torch
from tqdm import tqdm
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from utils.metrics import compute_eer

@dataclass
class EvaluationResults:
    eer: float
    threshold: float
    scores: List[Tuple[str, float]]
    labels: List[Tuple[str, int]]

class Evaluator:
    def __init__(self, config, model, test_loader):
        """Initialize the evaluator with model and data loader.

        Args:
            config: Configuration object containing evaluation parameters
            model: Neural network model to evaluate
            test_loader: DataLoader containing test data
        """
        self.config = config
        self.model = model.to(config.device)
        self.test_loader = test_loader
        self.output_path = Path(config.output_path)

    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, List[str], List[int]]:
        """Prepare batch data for model inference.

        Args:
            batch: Tuple containing features, utterance IDs, and labels

        Returns:
            Processed features tensor and original utterance IDs and labels
        """
        features, utt_ids, batch_labels = batch
        features = features.unsqueeze(1).float().to(
            self.config.device,
            non_blocking=True
        )
        return features, utt_ids, batch_labels

    def _collect_predictions(self) -> Tuple[List[float], List[int], List[str]]:
        """Collect model predictions for all test data.

        Returns:
            Tuple of (scores, labels, utterances)
        """
        scores = []
        labels = []
        utterances = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluation'):
                features, utt_ids, batch_labels = self._prepare_batch(batch)

                try:
                    _, outputs = self.model(features)
                    batch_scores = outputs[:, 1].cpu().numpy()

                    scores.extend(batch_scores)
                    labels.extend(batch_labels)
                    utterances.extend(utt_ids)
                    
                except RuntimeError as e:
                    print(f"Error processing batch: {e}")
                    continue

        return scores, labels, utterances

    def _compute_metrics(
        self,
        scores: List[float],
        labels: List[int],
        utterances: List[str]
    ) -> EvaluationResults:
        """Compute evaluation metrics from predictions.

        Args:
            scores: Model prediction scores
            labels: Ground truth labels
            utterances: Utterance IDs

        Returns:
            EvaluationResults object containing metrics and predictions
        """
        eer, threshold = compute_eer(scores, labels)
        
        print(type(scores))
	
        return EvaluationResults(
            eer=float(eer),
            threshold=float(threshold),
            scores=list(zip(utterances, scores)),
            labels=list(zip(utterances, labels))
        )

    def _save_results(self, results: EvaluationResults) -> None:
        """Save evaluation results to JSON file.

        Args:
            results: EvaluationResults object to save
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        

        results_dict = {
            'eer': results.eer,
            'threshold': results.threshold,
            'scores': results.scores,
            'labels': results.labels
        }

        try:
            with open(self.output_path / 'eval_results.json', 'w') as f:
                json.dump(str(results_dict), f, indent=2)
        except IOError as e:
            print(f"Error saving results: {e}")
            raise

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation process and return results.

        Returns:
            Dictionary containing evaluation metrics and predictions
        """
        scores, labels, utterances = self._collect_predictions()
        results = self._compute_metrics(scores, labels, utterances)
        self._save_results(results)

        return vars(results)
