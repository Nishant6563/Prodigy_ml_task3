# SVM Image Classification: Cats and Dogs
This repository provides a comprehensive implementation of a Support Vector Machine (SVM) to classify images of cats and dogs using the popular Kaggle Cats and Dogs dataset. The project includes data preprocessing, model training, evaluation, and saving the trained model.
# Table of Contents
1.Overview
2.Dataset
3.Requirements
4.Installation
5.Usage
6.Model Training
7.Evaluation
8.Saving the Model
9.Results
10.Contributing
11.License
# Overview
This project demonstrates how to build and train a Support Vector Machine (SVM) classifier to distinguish between images of cats and dogs. The steps include:

Loading and preprocessing the dataset.
Extracting features from the images.
Training the SVM model.
Evaluating the model's performance.
Saving the trained model for future use.
# Dataset
The dataset used in this project is the Kaggle Cats and Dogs dataset. It contains 25,000 images of cats and dogs, labeled accordingly.
# Requirements
1.Python 3.7+
2.NumPy
3.pandas
4.scikit-learn
5.OpenCV
6.matplotlib
7.tqdm
# Installation
Clone the repository:

git clone https://github.com/yourusername/cat-dog-svm-classifier.git
cd cat-dog-svm-classifier

Install the required packages:

pip install -r requirements.txt
# Usage
Download and extract the dataset from Kaggle.
Place the extracted dataset in the data/ directory:
data/
├── train/
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── dog.0.jpg
│   ├── dog.1.jpg
│   └── ...

# Feature Extraction
Run the feature extraction script to process the images and extract features:

python feature_extraction.py

# Model Training
rain the SVM model using the preprocessed data:

python train_model.py

# Saving the Model
Save the trained model for future use:

import pickle

with open('model.sav', 'wb') as file:
    pickle.dump(model, file)
# Results
The results of the model evaluation will be displayed, including accuracy, precision, recall, and F1-score
