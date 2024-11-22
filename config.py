import yaml
from dataclasses import dataclass
import torch
import os

@dataclass
class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Data configs
        self.access_type = config['data']['access_type']
        self.feature_type = config['data']['feature_type']
        self.feature_length = config['data']['feature_length']
        self.padding = config['data']['padding']
        self.num_workers = config['data']['num_workers']

        # Model configs
        self.resnet_type = config['model']['resnet_type']
        self.enc_dim = config['model']['enc_dim']
        self.num_classes = config['model']['num_classes']

        # Training configs
        self.seed = config['training']['seed']
        self.num_epochs = config['training']['num_epochs']
        self.batch_size = config['training']['batch_size']
        self.lr = float(config['training']['learning_rate'])
        self.lr_decay = float(config['training']['lr_decay'])
        self.lr_decay_interval = float(config['training']['lr_decay_interval'])
        self.beta1 = float(config['training']['beta1'])
        self.beta2 = float(config['training']['beta2'])
        self.epsilon = float(config['training']['epsilon'])
        self.weight_decay = float(config['training']['weight_decay'])

        # Paths
        self.features_path = config['paths']['features_path']
        self.protocol_path = config['paths']['protocol_path']
        self.output_path = config['paths']['output_path']
        self.checkpoint_path = config['paths']['checkpoint_path']

        # Device
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # Logger
        self.log_dir = config['logger']['log_dir']

        # Create necessary directories
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
