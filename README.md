# 🐱 Cat or Human? - ML Model

A TensorFlow/Keras model that classifies images as either cats or humans.

## 📦 Model Files

- `cat_or_human_model.keras` - The trained model file
- `haarcascade_frontalface_default.xml` - Face detection cascade classifier

## 🚀 Usage

### Training Scripts

- `cat_vs_human.py` - Main training script
- `test_img.py` - Test the model on a single image
- `webcam_face.py` - Real-time webcam detection script

### Dataset Structure

```
dataset/
├── train/
│   ├── cat/      # Cat training images
│   └── human/    # Human training images
└── test/
    ├── cat/      # Cat test images
    └── human/    # Human test images
```

## 📋 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🎯 Model Details

- **Input Size**: 224x224 pixels
- **Architecture**: Transfer learning model
- **Classes**: Cat (0) or Human (1)

## 🧪 Testing

Test on a single image:
```bash
python test_img.py
```

Run webcam detection:
```bash
python webcam_face.py
```

## 📝 Training

Train the model:
```bash
python cat_vs_human.py
```
