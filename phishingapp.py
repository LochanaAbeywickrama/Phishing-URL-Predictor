import streamlit as st
import numpy as np
import pandas as pd
import pickle
from urllib.parse import urlparse
import re
import tldextract
import requests
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io

# Load the dataset and preprocess
data = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')

# Drop non-relevant column
data = data.drop('FILENAME', axis=1)

# Convert all categorical data into numeric form to make machine-readable
from sklearn.preprocessing import LabelEncoder

# Define the categorical features
categorical_features = ['URL', 'Domain', 'TLD', 'Title']

# Create an instance of LabelEncoder
encoder = LabelEncoder()

# Encode each categorical feature separately
for feature in categorical_features:
    data[feature] = encoder.fit_transform(data[feature])

from sklearn.preprocessing import MinMaxScaler

# Normalization
scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

# Features and target variable (top 9 important features based on your analysis)
X = data[['IsHTTPS', 'NoOfSubDomain', 'HasSocialNet', 'HasCopyrightInfo', 'HasDescription', 
          'HasFavicon', 'NoOfPopup']]
y = data['label']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# Check performance on training set (debugging)
y_train_pred = rf.predict(X_train_scaled)
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred)}")

# Save the model using pickle
with open('phishing_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Load the model using pickle
with open('phishing_model.pkl', 'rb') as f:
    rf = pickle.load(f)


# Extract features based on the URL
def extract_features(url):
    features = {}

    # HTTPS Check
    features['IsHTTPS'] = int(urlparse(url).scheme == 'https')

    # Number of Subdomains
    parsed_url = tldextract.extract(url)
    subdomain = parsed_url.subdomain
    features['NoOfSubDomain'] = subdomain.count('.') + 1 if subdomain else 0

    # Attempt to fetch HTML for further analysis
    try:
        response = requests.get(url, timeout=5)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        # Has Favicon: Check if a favicon link is present
        features['HasFavicon'] = int(bool(soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon')))

        # Count Pop-up Windows
        features['NoOfPopup'] = html_content.lower().count('window.open')

        # Has Social Network Links: Check for common social media links
        social_sites = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube']
        features['HasSocialNet'] = int(any(site in html_content.lower() for site in social_sites))

        # Has Copyright Info: Check for "copyright" keyword
        features['HasCopyrightInfo'] = int('copyright' in html_content.lower())

        # Has Meta Description: Check for presence of meta description
        features['HasDescription'] = int(bool(soup.find('meta', attrs={'name': 'description'})))

    except requests.RequestException:
        # If the request fails, set feature values to 0
        features['HasFavicon'] = 0
        features['NoOfPopup'] = 0
        features['HasSocialNet'] = 0
        features['HasCopyrightInfo'] = 0
        features['HasDescription'] = 0

    # Return extracted features
    return features

# Function to apply rule-based predictions for phishing (0) or legitimate (1)
def make_predictions(features):
    # Initialize phishing score
    phishing_score = 0

    # Evaluate each condition and adjust the phishing score
    if features['IsHTTPS'] == 0:
        phishing_score += 1
    if features['NoOfSubDomain'] > 2:  
        phishing_score += 1
    if features['HasFavicon'] == 0:
        phishing_score += 1
    if features['NoOfPopup'] > 2:  
        phishing_score += 1
    if features['HasSocialNet'] == 0:
        phishing_score += 1
    if features['HasCopyrightInfo'] == 0:
        phishing_score += 1
    if features['HasDescription'] == 0:
        phishing_score += 1

    # Final decision based on the phishing score
    if phishing_score >= 4:  
        return 'Phishing'
    else:
        return 'Legitimate'

# Streamlit App Interface
# Sidebar for navigation
st.sidebar.title("Phishing URL Detection")
page = st.sidebar.selectbox("Select a page", ["Home", "Data Visualization", "Prediction"])

# Home Page
if page == "Home":
    st.title('Phishing URL Predictor')
    st.write("""
        This application allows you to:
        - Detect if a URL is likely to be Phishing or Legitimate using rule-based predictions.
        - Visualize data related to phishing and legitimate URLs.
        - Use a machine learning model to enhance phishing detection.
    """)
    st.image('ph.gif')
    st.write("Here is a preview of the dataset:")
    st.dataframe(data.head())
    st.write(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")

# Data Visualization Page
elif page == "Data Visualization":
    st.title("Data Visualization")
    st.markdown("Explore the dataset and various visualizations.")

    # Feature Importance Bar Chart
    feature_importances = rf.feature_importances_

    # Create a DataFrame for easier plotting
    feature_names = X.columns  # Feature names from the dataset
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Sort the features by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot the bar chart
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # Plot Distribution of Phishing vs Legitimate URLs as a Pie Chart
    fig, ax = plt.subplots()
    label_counts = data['label'].value_counts()
    labels = ['Legitimate', 'Phishing']
    colors = ['#4CAF50', '#FF0000']  # Green for Legitimate, Red for Phishing
    ax.pie(label_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
    ax.set_title("Phishing vs Legitimate URL Distribution")
    st.pyplot(fig)

    # Select only the relevant features for the correlation heatmap
    selected_features = ['IsHTTPS', 'NoOfSubDomain', 'HasSocialNet', 'HasCopyrightInfo', 'HasDescription', 'HasFavicon', 'NoOfPopup']
    corr_data = data[selected_features]

    # Plot the correlation heatmap for the selected features
    fig, ax = plt.subplots()
    corr = corr_data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation Heatmap (Selected Features)")
    st.pyplot(fig)

    # Scatter Plot: Relationship between 'NoOfSubDomain' and 'NoOfPopup'
    fig, ax = plt.subplots()
    ax.scatter(data['NoOfSubDomain'], data['NoOfPopup'], color='blue', alpha=0.5)
    ax.set_title("Scatter Plot: Number of Subdomains vs Number of Popups")
    ax.set_xlabel("Number of Subdomains")
    ax.set_ylabel("Number of Popups")
    st.pyplot(fig)


# Prediction Page
elif page == "Prediction":

    st.title('Phishing URL Predictor')
    st.write("""
        Enter a URL to predict whether it is Legitimate or Phishing using predefined rules or a trained model. 
    """)

    # Display progress
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)

    # Predict using the trained Random Forest model on test data (or any sample)
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)  
    
    # Display accuracy with the proper formatting
    st.markdown(f"<h2 style='color: red;'>**Model Accuracy: {accuracy:.2f}**</h2>", unsafe_allow_html=True)

    st.write()
    st.write()

    url_input = st.text_input("Enter the URL")

    if st.button("Predict"):
        features = extract_features(url_input)
        st.write("Extracted Features:", features)  # Debugging: Display extracted features
        result = make_predictions(features)
    
        if result == 'Phishing':
            st.error("This URL is classified as **Phishing**.")
            st.markdown('<span style="color:red;">🔴 Phishing URL</span>', unsafe_allow_html=True)
        else:
            st.success("This URL is classified as **Legitimate**.")
            st.markdown('<span style="color:green;">🟢 Legitimate URL</span>', unsafe_allow_html=True)
