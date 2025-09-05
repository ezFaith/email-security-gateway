# This script trains a machine learning model to detect phishing emails.
# It uses a larger, more realistic dataset to improve accuracy.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

print("Starting model training script with the new dataset...")

# Step 1: Load the new dataset (emails.csv)
try:
    # Load the new, larger dataset
    file_path = 'emails.csv'
    df = pd.read_csv(file_path, encoding='latin-1')
    print("Dataset loaded successfully.")
    
    # Print the columns to help with debugging if the names are different
    print("Columns in the loaded dataset:", df.columns.tolist())
    
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory.")
    exit()

# Step 2: Preprocess the data
# The dataset has 'label' and 'text' columns. We will explicitly rename and map the labels.
df.columns = ['is_phishing', 'email_text']

# Handle any rows with missing data (NaN) or empty text
df.dropna(subset=['email_text'], inplace=True)
df = df[df['email_text'].str.strip() != '']
print("Removed rows with missing or empty email text.")

# Convert 'Spam' labels to 1 (phishing) and 'Ham' labels to 0 (safe)
# This is a more robust method than a simple string search.
label_mapping = {'Spam': 1, 'Ham': 0}
df['is_phishing'] = df['is_phishing'].map(label_mapping)

# Check for a balanced dataset
phishing_count = df[df['is_phishing'] == 1].shape[0]
safe_count = df[df['is_phishing'] == 0].shape[0]

if phishing_count == 0 or safe_count == 0:
    print("Warning: The dataset is severely imbalanced. The model may perform poorly.")

print(f"Dataset has {safe_count} safe emails and {phishing_count} phishing emails.")

# Prepare data for training
X = df['email_text']
y = df['is_phishing']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split into training and testing sets.")

# Step 3: Vectorize the email text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("Text vectorized successfully.")

# Step 4: Train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# Step 5: Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))

# Step 6: Save the trained model and vectorizer
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(model, os.path.join('models', 'phishing_model.pkl'))
joblib.dump(tfidf_vectorizer, os.path.join('models', 'tfidf_vectorizer.pkl'))
print(f"Model and TfidfVectorizer saved as 'models/phishing_model.pkl' and 'models/tfidf_vectorizer.pkl'.")
print("Training script finished. You can now run 'app.py' after this.")
