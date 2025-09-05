import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)

# --- Configuration ---
DATASET_PATH = 'emails.csv'
MODELS_DIR = 'models'
TARGET_SAMPLES = 70000 # Use a subset of the dataset to reduce model size

# Ensure the models directory exists
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# --- Step 1: Load and Preprocess the Dataset ---
logging.info("Starting model training script with the new dataset...")

try:
    df = pd.read_csv(DATASET_PATH)
    logging.info("Dataset loaded successfully.")
except FileNotFoundError:
    logging.error(f"Error: The file '{DATASET_PATH}' was not found.")
    logging.info("Training script finished. Please ensure the dataset file is in the correct directory.")
    exit()

# Get a random sample of the data to keep model size small
if len(df) > TARGET_SAMPLES:
    df = df.sample(n=TARGET_SAMPLES, random_state=42)
    logging.info(f"Using a random sample of {TARGET_SAMPLES} emails to train the model.")

# Rename columns for consistency
df.rename(columns={'text': 'email', 'label': 'is_phishing'}, inplace=True)

# Drop rows where email text is missing or empty
df.dropna(subset=['email'], inplace=True)
df = df[df['email'].str.strip() != '']

# Map labels to numerical values (0 for safe, 1 for phishing)
df['is_phishing'] = df['is_phishing'].apply(lambda x: 1 if x.lower().strip() == 'spam' else 0)

logging.info(f"Dataset has {len(df[df['is_phishing'] == 0])} safe emails and {len(df[df['is_phishing'] == 1])} phishing emails.")

# Split the dataset into features (X) and target (y)
X = df['email']
y = df['is_phishing']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logging.info("Data split into training and testing sets.")

# --- Step 2: Feature Extraction ---
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit features to reduce model size
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
logging.info("Text vectorized successfully.")

# --- Step 3: Train the Classifier ---
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_tfidf, y_train)
logging.info("Model training complete.")

# --- Step 4: Evaluate the Model ---
y_pred = model.predict(X_test_tfidf)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))

# --- Step 5: Save the Trained Model and Vectorizer ---
joblib.dump(model, os.path.join(MODELS_DIR, 'phishing_model.pkl'))
joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))

logging.info(f"Model and TfidfVectorizer saved as 'models/phishing_model.pkl' and 'models/tfidf_vectorizer.pkl'.")
logging.info("Training script finished. You can now run 'app.py' after this.")