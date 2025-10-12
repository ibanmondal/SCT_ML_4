# Hand Gesture Recognition

A machine learning project that uses a Convolutional Neural Network (CNN) to recognize hand gestures from images. The project includes scripts for training the model on a dataset of hand gesture images and a real-time demo application that uses a webcam to predict gestures.

## Features

- **Model Training**: Train a CNN model on hand gesture images using data augmentation.
- **Real-time Recognition**: Use a webcam to capture live video and predict hand gestures in real-time.
- **Pre-trained Model**: Includes a pre-trained model (`hand_gesture_model.h5`) for immediate use.
- **Dataset**: Organized dataset with training and testing images for 20 different gestures.

## Installation

1. Clone or download the project repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The requirements include:
   - TensorFlow 2.13.0
   - OpenCV-Python
   - NumPy
   - Scikit-learn
   - Matplotlib

## Usage

### Training the Model

To train the model from scratch:

```bash
python main.py
```

This script will:
- Load images from the `train/train` directory.
- Apply data augmentation.
- Train the CNN model for 20 epochs.
- Save the trained model as `hand_gesture_model.h5`.
- Save gesture labels as `gesture_labels.npy`.
- Evaluate the model on test data from `test/test`.

### Real-time Demo

To run the real-time hand gesture recognition demo:

```bash
python demo.py
```

This will:
- Load the pre-trained model and labels.
- Open your webcam.
- Display the video feed with predicted gesture labels.
- Press 'q' to quit.

**Note**: Ensure your webcam is accessible. The demo processes frames in grayscale and resizes to 64x64 pixels.

## Dataset

The dataset consists of grayscale images of hand gestures, organized into folders for each gesture class (0-19).

- **Training Data**: Located in `train/train/`, with subfolders for each gesture.
- **Testing Data**: Located in `test/test/`, with subfolders for each gesture.

Each gesture folder contains multiple images. The model is trained to classify 20 different gestures.

## Model Architecture

The CNN model is defined in `model_utils.py` and consists of:

- 3 Convolutional layers with ReLU activation and MaxPooling.
- Flatten layer.
- Dense layer with 128 units and ReLU.
- Dropout (0.5) for regularization.
- Output Dense layer with softmax activation for multi-class classification.

Input shape: (64, 64, 1) - Grayscale images resized to 64x64.

## Files Description

- `main.py`: Script for loading data, training the model, and evaluation.
- `demo.py`: Real-time gesture recognition using webcam.
- `model_utils.py`: Functions to build, save, and load the CNN model.
- `requirements.txt`: List of Python dependencies.
- `hand_gesture_model.h5`: Pre-trained Keras model file.
- `gesture_labels.npy`: NumPy array of gesture class labels.
- `train/` and `test/`: Directories containing dataset images.

## Requirements

- Python 3.x
- Webcam (for demo)
- Sufficient RAM for training (depending on dataset size)

## License

This project is open-source. Feel free to use and modify.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for improvements.
