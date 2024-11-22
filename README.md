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