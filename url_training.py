# This script trains a machine learning model to detect phishing URLs.
# It reads a dataset, performs feature engineering on URLs, and saves the model.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import re
from urlextract import URLExtract

print("Starting URL model training script...")

# Step 1: Load the URL dataset
try:
    file_path = 'dataset_phishing.csv'
    df = pd.read_csv(file_path, encoding='latin-1')
    print("URL dataset loaded successfully.")
    
    # Verify columns and handle potential case differences
    df.columns = df.columns.str.strip().str.lower()
    
    # Check if the required columns exist
    if 'url' not in df.columns or 'status' not in df.columns:
        print("Error: The CSV file must contain 'url' and 'status' columns. Please check your file.")
        print(f"Columns found: {df.columns.tolist()}")
        exit()
    
    df.rename(columns={'url': 'URL', 'status': 'Label'}, inplace=True)
    
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory.")
    exit()

# Step 2: Feature Engineering
# Create new features from the URL string
def get_features(url):
    features = {}
    
    # 1. URL Length
    features['url_length'] = len(url)
    
    # 2. Number of dots
    features['num_dots'] = url.count('.')
    
    # 3. Presence of '@' symbol
    features['at_symbol'] = '@' in url
    
    # 4. Presence of '-' (hyphen)
    features['has_hyphen'] = '-' in url
    
    # 5. Check for IP address in URL
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    features['is_ip_based'] = bool(re.search(ip_pattern, url))
    
    # 6. Check for HTTPS protocol (a sign of legitimacy, but not foolproof)
    features['is_https'] = url.startswith('https')
    
    # 7. Number of subdomains
    domain_parts = url.split('//')[-1].split('/')[0].split('.')
    features['num_subdomains'] = len(domain_parts) - 2 # Subtracting domain and TLD
    
    return features

# Apply feature engineering to the dataset
feature_df = pd.DataFrame([get_features(url) for url in df['URL']])
full_df = pd.concat([df, feature_df], axis=1)

print("URL features extracted successfully.")

# Step 3: Prepare data for training
X = full_df[['url_length', 'num_dots', 'at_symbol', 'has_hyphen', 'is_ip_based', 'is_https', 'num_subdomains']]
y = full_df['Label']

# Encode the labels ('phishing' and 'legitimate') to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Original labels: {list(label_encoder.classes_)}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print("Data split into training and testing sets.")

# Step 4: Train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 6: Save the trained model and label encoder
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(model, os.path.join('models', 'url_model.pkl'))
joblib.dump(label_encoder, os.path.join('models', 'url_label_encoder.pkl'))
print(f"URL model and label encoder saved as 'models/url_model.pkl' and 'models/url_label_encoder.pkl'.")
print("Training script finished.")
