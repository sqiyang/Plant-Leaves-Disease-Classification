# Plant Leaves Disease Classification

This project implements a deep learning model for plant disease classification using transfer learning with ResNet50 on the PlantVillage dataset. It achieves 99.69% accuracy in identifying 38 classes of healthy and diseased leaves across 14 plant species.

## Project Overview

This project uses transfer learning with ResNet50 to classify 38 classes of healthy and diseased leaves from 14 plant species in the PlantVillage dataset (54,305 images). Key features:
- **Model**: ResNet50, fine-tuned with a custom fully connected layer.
- **Data Augmentation**: Random crops, flips, rotations, color jitter, affine transforms, and erasing.
- **Training**: Cross Entropy Loss with Adam optimizer (lr=0.0001), StepLR scheduler, and early stopping (patience=5).
- **Evaluation**: Achieves 99.69% test accuracy, with precision, recall, and F1-score near 1.0.

## Repository Contents

- **code/**: Jupyter Notebook (`code.ipynb`) for preprocessing, training, and evaluation.
- **data/**: Instructions for downloading PlantVillage dataset.
- **figures/**: Training and validation plots:
  - ![Accuracy Plot](figures/accuracy_plot.png)  
    Training and validation accuracy over epochs.
  - ![Loss Plot](figures/loss_plot.png)  
    Training and validation loss over epochs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PlantDiseaseClassifier.git
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy matplotlib pandas sklearn
   ```
3. Download the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset?select=color) and place it in `data/`.

## Usage

Run the Jupyter Notebook (`code.ipynb`) to:
1. Preprocess data and apply augmentations.
2. Train the model (ResNet50 with Adam optimizer).
3. Evaluate performance and generate plots.

Example command to start Jupyter:
```bash
jupyter notebook code.ipynb
```

## Results

- **Accuracy**: 99.69% on test set (Cross Entropy + Adam).
- **Metrics**: Precision, recall, and F1-score near 1.0 for most classes.
- **Plots**: Training/validation accuracy and loss in `figures/`.

## Future Work

- Add real-world image testing.
- Optimize for mobile deployment.
- Explore EfficientNet or Vision Transformers.





