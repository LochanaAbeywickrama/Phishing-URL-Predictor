# Phishing-URL-Predictor

This Streamlit application detects the state of a phishing URL with different features extracted from the URLs.

An interactive interface for visualizing distributions of features and predicting new URLs is provided.

**Features:**

Home: This is like an app introduction, giving basic information and the user can upload an image relevant to the phishing.

Data Visualization: How different features are distributed in a dataset like phishing vs legitimate URLs, number of subdomains, and feature importance.

Prediction: It will predict whether the URL is Phishing or Legitimate with a machine learning model.

According to the dataset, the prediction was done by the website URL using a Random Forest Classifier with 0.65 model accuracy. 

Example URLs to Test:

Phishing Example: http://123.456.789.000/ (Contains IP Address)

Legitimate Example: https://www.example.com/ (No phishing indicators)
