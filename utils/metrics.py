import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from typing import Tuple, Union, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EERResult:
    """Container for EER computation results."""
    eer: float
    threshold: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray

    def to_dict(self) -> dict:
        """Convert results to dictionary format."""
        return {
            'eer': float(self.eer),
            'threshold': float(self.threshold),
            'fpr': self.fpr.tolist(),
            'tpr': self.tpr.tolist(),
            'thresholds': self.thresholds.tolist()
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            json.dump(self.to_dict(), f, indent=2)

class EERComputer:
    """Class for computing Equal Error Rate (EER) metrics."""

    def __init__(self, pos_label: int = 1):
        """
        Initialize EER computer.

        Args:
            pos_label: Label to treat as positive class (default: 1)
        """
        self.pos_label = pos_label

    def _validate_inputs(self,
                        scores: np.ndarray,
                        labels: np.ndarray) -> None:
        """
        Validate input data for EER computation.

        Args:
            scores: Model prediction scores
            labels: Ground truth labels

        Raises:
            ValueError: If inputs are invalid
        """
        if len(scores) != len(labels):
            raise ValueError("Scores and labels must have the same length")

        if len(scores) == 0:
            raise ValueError("Input arrays cannot be empty")

        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("Labels must be binary (0 or 1)")

        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            raise ValueError("Scores contain NaN or Inf values")

    def _prepare_data(self,
                     scores: Union[List[float], np.ndarray],
                     labels: Union[List[int], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and validate input data.

        Args:
            scores: Raw prediction scores
            labels: Raw ground truth labels

        Returns:
            Tuple of processed (scores, labels)
        """
        scores_array = np.asarray(scores, dtype=np.float64)
        labels_array = np.asarray(labels, dtype=np.int32)

        self._validate_inputs(scores_array, labels_array)
        return scores_array, labels_array

    def _compute_roc_metrics(self,
                           scores: np.ndarray,
                           labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve metrics.

        Args:
            scores: Processed prediction scores
            labels: Processed ground truth labels

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        try:
            fpr, tpr, thresholds = roc_curve(
                labels,
                scores,
                pos_label=self.pos_label
            )
            return fpr, tpr, thresholds
        except Exception as e:
            logger.error(f"Error computing ROC curve: {e}")
            raise

    def _find_eer(self,
                  fpr: np.ndarray,
                  tpr: np.ndarray,
                  thresholds: np.ndarray) -> Tuple[float, float]:
        """
        Find the EER point from ROC metrics.

        Args:
            fpr: False positive rates
            tpr: True positive rates
            thresholds: Classification thresholds

        Returns:
            Tuple of (eer, threshold at eer)
        """
        if len(fpr) < 2 or len(tpr) < 2:
            raise ValueError("Insufficient points to compute EER")

        try:
            # Create interpolation functions
            tpr_interp = interp1d(fpr, tpr, kind='linear')
            thresh_interp = interp1d(fpr, thresholds, kind='linear',
                                   bounds_error=False,
                                   fill_value=(thresholds[0], thresholds[-1]))

            # Find EER using root finding
            eer = brentq(lambda x: 1. - x - tpr_interp(x), 0., 1.,
                        xtol=1e-8, rtol=1e-8)
            threshold = thresh_interp(eer)

            return float(eer), float(threshold)

        except Exception as e:
            logger.error(f"Error finding EER: {e}")
            raise

    def compute(self,
                scores: Union[List[float], np.ndarray],
                labels: Union[List[int], np.ndarray],
                return_full: bool = False) -> Union[Tuple[float, float], EERResult]:
        """
        Compute EER and related metrics.

        Args:
            scores: Prediction scores
            labels: Ground truth labels
            return_full: Whether to return full metrics

        Returns:
            If return_full is False: Tuple of (eer, threshold)
            If return_full is True: EERResult object

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If computation fails
        """
        try:
            # Prepare data
            scores_array, labels_array = self._prepare_data(scores, labels)

            # Compute ROC metrics
            fpr, tpr, thresholds = self._compute_roc_metrics(
                scores_array,
                labels_array
            )

            # Find EER point
            eer, threshold = self._find_eer(fpr, tpr, thresholds)

            if return_full:
                return EERResult(
                    eer=eer,
                    threshold=threshold,
                    fpr=fpr,
                    tpr=tpr,
                    thresholds=thresholds
                )
            return eer, threshold

        except Exception as e:
            logger.error(f"EER computation failed: {e}")
            raise

def compute_eer(scores: Union[List[float], np.ndarray],
                labels: Union[List[int], np.ndarray]) -> Tuple[float, float]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        scores: Prediction scores
        labels: Ground truth labels

    Returns:
        Tuple of (eer, threshold)
    """
    computer = EERComputer()
    return computer.compute(scores, labels)
