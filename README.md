# EmbeddingClassifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Image%20Classification-orange.svg)]()

## 📋 Table of Contents
- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## 🎯 About

EmbeddingClassifier is a deep learning project focused on **image classification using embedding techniques**. This project implements state-of-the-art embedding-based approaches to classify images efficiently and accurately. By leveraging learned embeddings, the classifier can capture semantic relationships between different image categories and provide robust classification performance.

The project is designed to be modular, extensible, and easy to use for both research and production environments.

## ✨ Features

- 🔥 **Embedding-based Classification**: Utilizes deep embedding techniques for robust image classification
- 🚀 **High Performance**: Optimized for both accuracy and inference speed
- 🛠️ **Modular Design**: Easy to extend and customize for different use cases
- 📊 **Multiple Metrics**: Comprehensive evaluation with various classification metrics
- 🎨 **Visualization Tools**: Built-in tools for visualizing embeddings and results
- 📱 **Easy Integration**: Simple API for integration into existing workflows
- 🔧 **Configurable**: Flexible configuration system for different scenarios
- 📈 **Training Pipeline**: Complete training pipeline with monitoring and checkpointing

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- GPU support (optional but recommended)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Rudra-iitg/EmbeddingClassifier.git
cd EmbeddingClassifier

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Quick Install (Coming Soon)
```bash
pip install embedding-classifier
```

## 💻 Usage

### Basic Usage

```python
from embedding_classifier import EmbeddingClassifier
from embedding_classifier.utils import load_image

# Initialize the classifier
classifier = EmbeddingClassifier(model_path='path/to/trained/model.pth')

# Load and classify an image
image = load_image('path/to/your/image.jpg')
prediction = classifier.predict(image)

print(f"Predicted class: {prediction['class']}")
print(f"Confidence: {prediction['confidence']:.4f}")
```

### Training a New Model

```python
from embedding_classifier import EmbeddingClassifier
from embedding_classifier.data import ImageDataLoader

# Prepare your dataset
train_loader = ImageDataLoader('path/to/train/data', batch_size=32)
val_loader = ImageDataLoader('path/to/val/data', batch_size=32)

# Initialize classifier
classifier = EmbeddingClassifier(
    embedding_dim=512,
    num_classes=10,
    architecture='resnet50'
)

# Train the model
classifier.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    learning_rate=0.001
)

# Save the trained model
classifier.save('models/my_classifier.pth')
```

### Advanced Configuration

```python
# Custom configuration
config = {
    'model': {
        'architecture': 'efficientnet-b0',
        'embedding_dim': 256,
        'dropout_rate': 0.3
    },
    'training': {
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'weight_decay': 0.01
    },
    'data': {
        'image_size': 224,
        'augmentation': True,
        'normalization': 'imagenet'
    }
}

classifier = EmbeddingClassifier(config=config)
```

## 📁 Project Structure

```
EmbeddingClassifier/
├── embedding_classifier/          # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── base_model.py         # Base model class
│   │   ├── embedding_net.py      # Embedding network
│   │   └── classifier.py         # Classification head
│   ├── data/                     # Data handling modules
│   │   ├── __init__.py
│   │   ├── datasets.py           # Dataset classes
│   │   ├── transforms.py         # Data transformations
│   │   └── loaders.py            # Data loaders
│   ├── training/                 # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop
│   │   ├── losses.py             # Loss functions
│   │   └── metrics.py            # Evaluation metrics
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── visualization.py      # Plotting and visualization
│   │   ├── config.py             # Configuration handling
│   │   └── helpers.py            # Helper functions
│   └── inference/                # Inference utilities
│       ├── __init__.py
│       ├── predictor.py          # Prediction interface
│       └── export.py             # Model export utilities
├── examples/                     # Example scripts and notebooks
│   ├── basic_usage.py            # Basic usage example
│   ├── training_example.py       # Training example
│   └── notebooks/                # Jupyter notebooks
│       └── tutorial.ipynb        # Tutorial notebook
├── tests/                        # Test files
│   ├── __init__.py
│   ├── test_models.py            # Model tests
│   ├── test_data.py              # Data handling tests
│   └── test_training.py          # Training tests
├── configs/                      # Configuration files
│   ├── default.yaml              # Default configuration
│   └── experiments/              # Experiment configurations
├── docs/                         # Documentation
│   ├── api_reference.md          # API documentation
│   └── tutorials/                # Tutorial documentation
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── README.md                     # This file
└── LICENSE                       # License file
```

## 📋 Requirements

### Core Dependencies
- `torch >= 1.8.0`
- `torchvision >= 0.9.0`
- `numpy >= 1.19.0`
- `pillow >= 8.0.0`
- `matplotlib >= 3.3.0`
- `scikit-learn >= 0.24.0`
- `tqdm >= 4.60.0`

### Optional Dependencies
- `tensorboard >= 2.5.0` (for training visualization)
- `wandb >= 0.12.0` (for experiment tracking)
- `opencv-python >= 4.5.0` (for advanced image processing)

See `requirements.txt` for the complete list of dependencies.

## 🏗️ Model Architecture

The EmbeddingClassifier uses a two-stage approach:

1. **Embedding Network**: Extracts meaningful feature embeddings from input images
   - Supports various backbone architectures (ResNet, EfficientNet, Vision Transformer)
   - Configurable embedding dimensions
   - Optional batch normalization and dropout

2. **Classification Head**: Maps embeddings to class predictions
   - Fully connected layers with configurable depth
   - Supports multiple activation functions
   - Built-in regularization techniques

### Supported Architectures
- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0-B7)
- Vision Transformer (ViT)
- MobileNet V2/V3
- DenseNet

## 📊 Dataset

The classifier expects datasets organized in the following structure:

```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    │   ├── image1.jpg
    │   └── ...
    └── class2/
        ├── image1.jpg
        └── ...
```

### Supported Formats
- Image formats: JPG, PNG, BMP, TIFF
- Automatic resizing and normalization
- Built-in data augmentation options

## 🎓 Training

### Training Script Example

```bash
python scripts/train.py \
    --data_path /path/to/dataset \
    --model_config configs/resnet50.yaml \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --output_dir models/
```

### Training Features
- Automatic mixed precision training
- Learning rate scheduling
- Early stopping
- Model checkpointing
- TensorBoard integration
- Multi-GPU support

## 📈 Evaluation

### Metrics
- Accuracy (Top-1, Top-5)
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC (for binary classification)
- Embedding visualization (t-SNE, UMAP)

### Evaluation Script

```bash
python scripts/evaluate.py \
    --model_path models/best_model.pth \
    --data_path /path/to/test/dataset \
    --output_dir results/
```

## 🎨 Examples

### 1. Quick Classification
```python
from embedding_classifier import EmbeddingClassifier

# Load pre-trained model
classifier = EmbeddingClassifier.load_pretrained('imagenet-resnet50')

# Classify image
result = classifier.classify('example.jpg')
print(f"Class: {result.class_name}, Confidence: {result.confidence}")
```

### 2. Batch Processing
```python
import glob
from embedding_classifier import EmbeddingClassifier

classifier = EmbeddingClassifier.load('models/my_model.pth')

# Process multiple images
image_paths = glob.glob('images/*.jpg')
results = classifier.classify_batch(image_paths, batch_size=16)

for path, result in zip(image_paths, results):
    print(f"{path}: {result.class_name} ({result.confidence:.3f})")
```

### 3. Custom Dataset Training
See `examples/training_example.py` for a complete training example with custom datasets.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone the repository
git clone https://github.com/Rudra-iitg/EmbeddingClassifier.git
cd EmbeddingClassifier

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Rudra Jha** - [@Rudra-iitg](https://github.com/Rudra-iitg)

Project Link: [https://github.com/Rudra-iitg/EmbeddingClassifier](https://github.com/Rudra-iitg/EmbeddingClassifier)

## 🙏 Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- Inspired by various embedding-based classification research papers
- Special thanks to the open-source community for valuable contributions

---

⭐ **Star this repository if you find it helpful!**
