# 🐱 Cat or Human? - ML Classification Model

A machine learning project that uses transfer learning with TensorFlow/Keras to classify images as either cats or humans. The model leverages MobileNetV2 architecture for efficient and accurate binary classification.

## 📦 Model Files

- `cat_or_human_model.keras` - The trained TensorFlow/Keras model file
- `haarcascade_frontalface_default.xml` - OpenCV Haar cascade classifier for face detection (included for potential future enhancements)

## 🚀 Quick Start

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

Ensure your dataset follows this structure:
```
dataset/
├── train/
│   ├── cat/      # Training images of cats
│   └── human/    # Training images of humans
└── test/
    ├── cat/      # Test images of cats
    └── human/    # Test images of humans
```

## 🏃‍♂️ Usage

### Training the Model

Train a new model from scratch:
```bash
python cat_vs_human.py
```

**Training Details:**
- Uses MobileNetV2 as base model (pre-trained on ImageNet)
- Input size: 224x224 pixels
- Binary classification: Cat (0) or Human (1)
- Batch size: 32
- Default epochs: 5
- Data augmentation: horizontal flip and zoom

### Testing the Model

Test the trained model on a single image:
```bash
python test_img.py
```

**Note:** The test script currently loads `test2.jpg` from the project root. Update the script to test different images as needed.

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Pillow

## 🎯 Model Architecture

- **Base Model**: MobileNetV2 (transfer learning)
- **Input Shape**: (224, 224, 3)
- **Layers**: Global Average Pooling + Dense(128) + Dense(1, sigmoid)
- **Output**: Probability score (0-1), where >0.5 = Human, ≤0.5 = Cat

## 📊 Performance

The model's performance depends on:
- Quality and quantity of training data
- Image preprocessing and augmentation
- Number of training epochs
- Fine-tuning parameters

## 🤝 Contributing

Feel free to contribute by:
- Improving the model architecture
- Adding more data preprocessing techniques
- Enhancing the testing capabilities
- Adding support for batch testing

## 📄 License

This project is licensed under the terms specified in the LICENSE file.
