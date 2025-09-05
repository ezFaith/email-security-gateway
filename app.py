import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
import re
import os
from urlextract import URLExtract
import logging

# Configure logging to show all messages
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder='templates')

# File paths for models
EMAIL_MODEL_PATH = os.path.join('models', 'phishing_model.pkl')
TFIDF_VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')
URL_MODEL_PATH = os.path.join('models', 'url_model.pkl')
URL_LABEL_ENCODER_PATH = os.path.join('models', 'url_label_encoder.pkl')

# Global variables to store the loaded models
email_model = None
tfidf_vectorizer = None
url_model = None
url_label_encoder = None
url_extractor = URLExtract()

# Load models and vectorizers at application startup
def load_models():
    global email_model, tfidf_vectorizer, url_model, url_label_encoder
    
    # Load email detection models
    try:
        email_model = joblib.load(EMAIL_MODEL_PATH)
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        logging.info("Email detection models loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Error loading email models: {e}. Please run model_training.py first.")
        email_model = None
        tfidf_vectorizer = None

    # Load URL detection models
    try:
        url_model = joblib.load(URL_MODEL_PATH)
        url_label_encoder = joblib.load(URL_LABEL_ENCODER_PATH)
        logging.info("URL detection models loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Error loading URL models: {e}. Please run url_training.py first.")
        url_model = None
        url_label_encoder = None

# Function to get features from a URL, matching the training script
def get_url_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['at_symbol'] = '@' in url
    features['has_hyphen'] = '-' in url
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    features['is_ip_based'] = bool(re.search(ip_pattern, url))
    features['is_https'] = url.startswith('https')
    domain_parts = url.split('//')[-1].split('/')[0].split('.')
    features['num_subdomains'] = len(domain_parts) - 2
    return pd.DataFrame([features])

# Main route for the web interface
@app.route('/')
def home():
    # Attempt to load models before rendering the template
    load_models()
    return render_template('index.html')

@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    if not email_model or not tfidf_vectorizer or not url_model or not url_label_encoder:
        return jsonify({
            'status': 'error',
            'message': 'One or more models failed to load. Please check the logs and ensure training scripts were run.'
        })

    # The new frontend sends a JSON object in the request body
    data = request.get_json(silent=True)
    if not data or 'email_content' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Invalid request: No email content provided.',
            'reasons': []
        })

    email_text = data['email_content']
    reasons = []
    final_status = "Safe"
    url_phishing_count = 0

    # 1. Keyword-based Analysis
    phishing_keywords = [
        "verify account", "password reset", "security alert", "click here",
        "update your information", "urgently", "account suspended", "login now"
    ]
    keyword_matches = [word for word in phishing_keywords if word in email_text.lower()]
    if keyword_matches:
        reasons.append(f"Suspicious keywords found: {', '.join(keyword_matches)}.")
        final_status = "Phishing Detected"

    # 2. Machine Learning Model Analysis (Email Body)
    email_text_processed = tfidf_vectorizer.transform([email_text])
    email_prediction = email_model.predict(email_text_processed)[0]
    email_prediction_proba = email_model.predict_proba(email_text_processed)[0]
    
    confidence = email_prediction_proba[1] if email_prediction == 1 else email_prediction_proba[0]
    confidence_percent = round(confidence * 100, 0)

    if email_prediction == 1:
        reasons.append(f"Content analysis model predicts: Phishing (Confidence: {confidence_percent} %).")
        final_status = "Phishing Detected"
    else:
        reasons.append(f"Content analysis model predicts: Safe (Confidence: {confidence_percent} %).")
    
    # 3. URL Analysis
    urls_found = url_extractor.find_urls(email_text)
    if urls_found:
        reasons.append("URL Analysis:")
        for url in urls_found:
            # First, use your local ML model and rule-based check
            url_features = get_url_features(url)
            url_prediction = url_model.predict(url_features)[0]
            url_label = url_label_encoder.inverse_transform([url_prediction])[0]
            
            tld = url.split('//')[-1].split('/')[0].split('.')[-1]
            malicious_tlds = ['xyz', 'top', 'info', 'site', 'tk']
            is_malicious_tld = tld.lower() in malicious_tlds
            
            url_status_display = url_label.capitalize()
            if is_malicious_tld:
                url_status_display = "Phishing (Rule-Based)"
                url_phishing_count += 1
            elif url_label == 'phishing':
                url_status_display = "Phishing (Model-Based)"
                url_phishing_count += 1
            
            reasons.append(f"URL: {url} -> Status: {url_status_display}")
    
    # Final decision based on combined analysis
    if url_phishing_count > 0:
        reasons.append(f"Final Conclusion: {url_phishing_count} suspicious URL(s) detected. ")
        final_status = "Phishing Detected"

    response = {
        'status': final_status,
        'message': 'Analysis Complete',
        'reasons': reasons
    }
    
    return jsonify(response)

if __name__ == '__main__':
    load_models()
    app.run(debug=True)