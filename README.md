# Breed_classification

## Overview

This project focuses on classifying dog breeds using deep learning techniques. The goal is to build an effective model capable of distinguishing between various dog breeds. The project encompasses data preprocessing, model training, hyperparameter tuning, and evaluation.

## Project Structure

The project is organized into several key sections:

### 1. Import Modules and Dataset

- Various Python modules are imported for data manipulation, visualization, and deep learning.
- The project uses a dog breed classification dataset from Stanford University.

### 2. Helper Functions

- Helper functions are defined for visualizing images, plotting training history, benchmarking models, showing results, and generating a classification report.

### 3. Data Preprocessing and Visualization

- The dataset is downloaded and extracted.
- Images are cropped and preprocessed to reduce background noise.
- Data is split into training, validation, and test sets.
- Image data generators are created for data augmentation.

### 4. Simple Model CNN from Scratch as a Baseline

- A simple CNN model is implemented as a baseline for dog breed classification.
- The model is trained, and its performance is visualized.

### 5. Transfer Learning with NASNetLarge

- NASNetLarge, a pre-trained deep learning model, is used for transfer learning.
- The model is fine-tuned on the dog breed dataset, and its performance is evaluated.

### 6. Hyperparameter Tuning with Keras Tuner

- Hyperparameters for the NASNetLarge model are optimized using Keras Tuner.
- The best hyperparameters are selected, and the model is re-trained.

### 7. Fine-Tuning NASNetLarge

- The last layers of the NASNetLarge model are fine-tuned to improve performance.
- The fine-tuned model is evaluated on the test set.

### 8. Making Predictions and Analysis

- Predictions are made using the trained models.
- Results are visualized, and misclassifications are analyzed.
- A confusion matrix and classification report are generated.

### 9. Time Duration

- The total time taken for the project execution is displayed at the end.

### 10. Saved Models and Classes

- The final models (NASNetLarge and fine-tuned NASNetLarge) are saved.
- The class labels are saved using pickle for future use.

**Note:** Ensure that the necessary datasets and dependencies are available before running the code.

## Author

This project was created by Fouad Maherzi.
