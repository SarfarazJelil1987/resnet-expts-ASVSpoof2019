# Audio Spoofing Detection System for ASVspoof 2019
## Project Description
This project implements a deep learning-based system for detecting 
genuine and spoofed speech as part of the ASVspoof 2019 challenge. The system utilizes a
ResNet architecture to discriminate between genuine and artificially generated speech, 
helping to protect automatic speaker verification (ASV) systems against spoofing attacks.

## Key Features
- ResNet-based deep learning architecture
- Performance evaluation using Equal Error Rate (EER)
- Configurable model parameters via YAML
- Efficient data loading and processing pipeline

## Dependencies
### Required Packages
```bash
torch==1.12.1
torchaudio==0.13.1
numpy==1.24.1
scipy==1.10.0
PyYAML==5.3.1
tensorboard==2.11.2
tqdm==4.64.1
scikit-learn==1.2.0
```

## Get Started
### Installation
Create a new virtual environment and install the required packages:
```bash
python3 -m venv asvspoof_env
source asvspoof_env/bin/activate 
pip install -r requirements.txt
```

### Project Structure
```bash
resnet-expts-ASVSpoof2019/
├── configs/
│   └── config.yaml      # Configuration parameters
├── data/
│   └── dataloader.py    # Data loading and preprocessing
│    └── dataset.py      # Dataset class for ASVSpoof 2019 data
├── model/
│   └── resnet.py        # Model architecture definition
├── trainer/
│   └── evaluator.py     # Evaluation metrics and testing
│   └── trainer.py       # Trainer class 
├── train.py             # Main training script
├── test.py              # Main testing script
└── requirements.txt     # Project dependencies
```

## Configuration
The system parameters are configured through configs/config.yaml:

## Model Configuration
```text
enc_dim: 256            # Encoder dimension
resnet_type: "resnet18" # ResNet architecture type
num_classes: 2          # Number of output classes
```
## Training Configuration
```text
checkpoint_path: "checkpoints/"  # Model checkpoint directory
```

## Data Configuration
```textmate
data_path: "path/to/data"       # Dataset location
batch_size: 32                  # Batch size for evaluation
```

## Running the System

### Data Preparation
Download the ASVspoof 2019 dataset and extract the 2D-ILRCC features (or any feature)
in the pickle format(.pkl).

Update the feature path in config.yaml.

Ensure the data is organized according to the ASVspoof 2019 protocol.

## Model Training
Train the model using the training script:
```bash
python3 train.py
```
## Model Evaluation
Run the evaluation script:

```bash
python3 test.py
```

## Model Architecture
The system uses a ResNet-based architecture with:

- Input layer for processing audio features

- Configurable number of ResNet blocks

- Encoder dimension: 256 (configurable)

- Output layer for binary classification (genuine/spoof)

## Performance Optimization
For optimal performance:

- Use GPU acceleration when available

- Adjust batch size based on available memory

- Ensure proper data preprocessing

## Troubleshooting
Common issues and solutions:

- CUDA out of memory: Reduce batch size in config.yaml

- Model loading errors: Verify checkpoint path and model architecture match

## Citation
If you use this code in your research, please cite:
```bibtex
@ARTICLE{10449475,
  author={Jelil, Sarfaraz and Sinha, Rohit and Prasanna, S. R. Mahadeva},
  journal={IEEE Signal Processing Letters}, 
  title={Spectro-Temporally Compressed Source Features for Replay Attack Detection}, 
  year={2024},
  volume={31},
  number={},
  pages={721-725}, 
  doi={10.1109/LSP.2024.3370490}}
```

## Contact
```html
safaraz@iitg.ac.in
```


## Contributing
- Fork the repository

- Create a feature branch

- Commit changes

- Push to the branch

- Create a Pull Request
