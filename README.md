# -Image-Classification
GitHub Repository Name: Intel-Image-Classifier  Description: A CNN-based image classifier for Intel Image Classification dataset. Classifies images into 6 categories: buildings, forest, glacier, mountain, sea, street. Built with TensorFlow/Keras.


## Overview:
This project implements a Convolutional Neural Network (CNN) for image classification using TensorFlow.

## Dataset:
- The dataset contains around 25,000 images of size 150x150 pixels.
- Images are distributed across six categories: buildings, forest, glacier, mountain, sea, and street.
- The dataset is divided into three sets: train, validation, and test.
  - Train set: Approximately 14,000 images.
  - Validation set: Approximately 7,000 images.
  - Test set: Approximately 3,000 images.
- Each image is labelled with the corresponding category (0 for buildings, 1 for forest, 2 for glacier, 3 for mountain, 4 for sea, and 5 for street).

## Model Development:
- Chosen model architecture: Convolutional Neural Network (CNN).
- Implemented using TensorFlow.
- Model architecture includes convolutional layers, pooling layers, fully connected layers, and softmax activation.

## Data Preprocessing:
- Image preprocessing:
  - Resizing images to 150x150 pixels.
  - Rescaling pixel values to the range [0, 1].
- Label preprocessing:
  - Encoding categorical labels to numerical format.

## Training and Evaluation:
- Trained the model on the training set.
- Evaluated the model's performance on the validation set.
- Optimization:
  - Experimented with different hyperparameters to improve performance.
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## Results:
- Trained model saved as `final_model.keras`.
- Evaluation results stored in `evaluation_results.txt`.

## Usage:
1. Clone the repository:
2. Install dependencies:
3. Download the dataset from (https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data?select=seg_train) and place it in the `data` directory.
4. Run `train.py` to train the model.
5. Run `evaluate.py` to evaluate the trained model on the validation set.



