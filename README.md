# Plant Leaves Disease Classification

This project implements a deep learning model for plant disease classification using transfer learning with ResNet50 on the PlantVillage dataset. It achieves 99.69% accuracy in identifying 38 classes of healthy and diseased leaves across 14 plant species.

## Project Overview

The goal is to detect and classify plant leaf diseases to support global food security and sustainable farming. The model leverages computer vision and deep learning, trained on 54,305 leaf images from the PlantVillage dataset, covering 14 plant species and 38 classes (healthy and diseased). Key techniques include:

- **Transfer Learning**: Pre-trained ResNet50 model, fine-tuned for 38-class classification.
- **Data Augmentation**: Random crops, flips, rotations, color jittering, affine transformations, and erasing to enhance generalization.
- **Training**: Uses Cross Entropy Loss with Adam optimizer and Multi Margin Loss with SGD, with early stopping to prevent overfitting.
- **Evaluation Metrics**: Accuracy (99.69% with Adam), precision, recall, and F1-score.

## Repository Contents

- **code/**: Python scripts for data preprocessing, model training, and evaluation.
- **report/**: Project report (`AIT2104016.pdf`) detailing methodology, results, and analysis.
- **data/**: Instructions for downloading the PlantVillage dataset.
- **figures/**: Visualizations of training/validation accuracy and loss.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PlantDiseaseClassifier.git
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy matplotlib pandas
   ```
3. Download the PlantVillage dataset and place it in the `data/` folder (see `data/README.md` for details).

## Usage

1. Preprocess the dataset:
   ```bash
   python code/preprocess.py
   ```
2. Train the model:
   ```bash
   python code/train.py --optimizer adam --loss cross_entropy
   ```
3. Evaluate the model:
   ```bash
   python code/evaluate.py
   ```

## Results

- **Cross Entropy Loss + Adam**: 99.69% test accuracy, with precision/recall/F1-score near 1.0 for most classes.
- **Multi Margin Loss + SGD**: 94.61% test accuracy, showing stable but less optimal performance.
- Visualizations of training/validation accuracy and loss are in the `figures/` directory.

## Future Work

- Expand dataset with real-world, noisy images.
- Optimize for mobile deployment.
- Explore newer architectures like EfficientNet or Vision Transformers.



