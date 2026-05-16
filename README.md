# Crop Recommendation System

A simple machine learning project that suggests the most suitable crop to grow based on soil nutrients and weather conditions.

The model is trained using a Random Forest Classifier on a dataset containing values for nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.

## How it works
The system learns patterns from past agricultural data. When new soil and climate values are given, it predicts the best crop that matches those conditions.

## Model used
Random Forest Classifier

## Input features
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH level
- Rainfall

## Output
Predicted crop name (for example: rice, wheat, maize, etc.)

## How to run
1. Install required libraries
   pip install pandas numpy scikit-learn

2. Run the script
   python crop_model.py

## Note
This is a basic machine learning project made for learning and practice.
# Crop-recommandation-ML-model
