# Handwritten Digit Recognition using Support Vector Machines

This project aims to recognize handwritten digits using Support Vector Machines (SVM) on the MNIST dataset.

## Project Overview

In this project, I've used the famous MNIST dataset, which contains 70,000 images of handwritten digits (0-9), to build a model that can correctly identify the digit in the image. I've employed Support Vector Machines (SVM), a powerful supervised learning algorithm for classification tasks.

## Project Steps

### 1. Data Collection

We download the MNIST dataset, which contains 70,000 images of handwritten digits, in CSV format. The dataset is divided into two files: mnist_train.csv (60,000 training examples) and mnist_test.csv (10,000 test examples).

Dataset: [MNIST Dataset on Kaggle](https://www.kaggle.com/oddrationale/mnist-in-csv)

### 2. Data Preprocessing

We preprocess the dataset by dividing it into training and testing sets. The training set is used to train the model, while the testing set is used to evaluate its performance. We split the features (pixel values) and labels (digits) accordingly.

### 3. Model Building

We build a Support Vector Machine (SVM) model using the scikit-learn library in Python. We choose the Radial Basis Function (RBF) kernel and set the regularization parameter C to 1.0.

### 4. Model Training

We train the SVM model on the training dataset using the `fit` method.

### 5. Model Evaluation

We evaluate the performance of the SVM model on the testing dataset by calculating metrics such as accuracy, precision, recall, and F1-score using scikit-learn's evaluation functions.

## Usage

To use the trained model for predicting handwritten digits from PNG images:

1. Save the trained SVM model using joblib or pickle.
2. Write a Python script to preprocess input images, load the saved model, and make predictions.
3. Deploy the script as an API on a cloud platform for real-time predictions.

## Files Included

- `predict.py`: Model evaluation on Python
- `app.py` : Flask backend
- `streamlit_app.py` : Streamlit frontend
- `mnist_test.csv`: Testing dataset containing 10,000 examples.
- `svm_model.pkl`: Saved SVM model.
- `README.md`: Project overview and usage instructions.

## Acknowledgements

- The MNIST dataset is sourced from Kaggle: [MNIST Dataset on Kaggle](https://www.kaggle.com/oddrationale/mnist-in-csv)
- We used scikit-learn for building and training the SVM model.

### Arya Suneesh - 2021BCS0005