from typing import Dict, List, Tuple, Union, Optional
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASVspoof2019Dataset(Dataset):
    """
    Dataset class for ASVspoof2019 data handling.

    Handles loading and preprocessing of ASVspoof2019 dataset features and labels.
    Supports different padding strategies and feature length configurations.

    Author: Sarfaraz Jelil
    """

    VALID_SPLITS = {'train', 'dev', 'eval'}
    VALID_PADDING = {'zero', 'repeat'}

    def __init__(self, config: object, split: str = 'train') -> None:
        """
        Initialize the dataset.

        Args:
            config: Configuration object containing:
                   - features_path: Path to features directory
                   - protocol_path: Path to protocol files
                   - access_type: Type of access ('LA' or 'PA')
                   - feature_type: Type of features
                   - feature_length: Length of features
                   - padding: Padding strategy ('zero' or 'repeat')
            split: Dataset split ('train', 'dev', or 'eval')

        Raises:
            ValueError: If invalid split or padding strategy is provided
            FileNotFoundError: If required directories or files are missing
        """
        self._validate_inputs(config, split)
        self.config = config
        self.split = split

        # Setup paths
        self.features_dir = self._setup_features_dir()
        self.protocol_data = self._load_protocol()

        logger.info(f"Initialized {split} dataset with {len(self.protocol_data)} samples")

    def _validate_inputs(self, config: object, split: str) -> None:
        """Validate initialization inputs."""
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Split must be one of {self.VALID_SPLITS}")

        if not hasattr(config, 'padding') or config.padding not in self.VALID_PADDING:
            raise ValueError(f"Padding must be one of {self.VALID_PADDING}")

        required_attrs = ['features_path', 'protocol_path', 'access_type',
                         'feature_type', 'feature_length']
        missing = [attr for attr in required_attrs if not hasattr(config, attr)]
        if missing:
            raise ValueError(f"Config missing required attributes: {missing}")

    def _setup_features_dir(self) -> Path:
        """Setup and validate features directory."""
        features_dir = Path(self.config.features_path) / \
                      f'{self.split}'

        if not features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {features_dir}")

        return features_dir

    def _load_protocol(self) -> List[Dict]:
        """
        Load and parse protocol file.

        Returns:
            List of dictionaries containing utterance IDs and labels

        Raises:
            FileNotFoundError: If protocol file doesn't exist
        """
        protocol_file = Path(self.config.protocol_path) / \
                       f'ASVspoof2019.{self.config.access_type}.cm.{self.split}.trl.txt'

        if not protocol_file.exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_file}")

        try:
            data = []
            with open(protocol_file, 'r') as f:
                for line in f:
                    _, utt_id, _, _, label = line.strip().split()
                    data.append({
                        'utt_id': utt_id,
                        'label': 1 if label == 'bonafide' else 0
                    })
            return data
        except Exception as e:
            logger.error(f"Error loading protocol file: {e}")
            raise

    def _load_feature(self, feat_path: Path) -> np.ndarray:
        """Load feature file and handle errors."""
        try:
            with open(feat_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading feature file {feat_path}: {e}")
            raise

    def _pad_feature(self, feat: np.ndarray) -> np.ndarray:
        """Apply padding to feature array."""
        target_length = self.config.feature_length

        if feat.shape[1] >= target_length:
            return feat[:, :target_length]

        if self.config.padding == 'repeat':
            return self._repeat_pad(feat, target_length)
        return self._zero_pad(feat, target_length)

    def _repeat_pad(self, feat: np.ndarray, target_length: int) -> np.ndarray:
        """Pad by repeating the feature."""
        return np.repeat(feat, np.ceil(target_length/feat.shape[1]), axis=1)[:, :target_length]

    def _zero_pad(self, feat: np.ndarray, target_length: int) -> np.ndarray:
        """Pad with zeros."""

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.protocol_data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing:
            - Feature array (padded if necessary)
            - Utterance ID
            - Label (1 for bonafide, 0 for spoof)
        """
        item = self.protocol_data[idx]
        feat_path = self.features_dir / f"{item['utt_id']}.pkl"

        feat = self._load_feature(feat_path)
        feat = self._pad_feature(feat)

        return feat, item['utt_id'], item['label']
