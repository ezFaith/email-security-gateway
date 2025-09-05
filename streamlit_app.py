import streamlit as st
import joblib
import pandas as pd
import re
import os

# --- Helper Functions (matching app.py and url_training.py) ---

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

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        email_model = joblib.load(os.path.join('models', 'phishing_model.pkl'))
        tfidf_vectorizer = joblib.load(os.path.join('models', 'tfidf_vectorizer.pkl'))
        url_model = joblib.load(os.path.join('models', 'url_model.pkl'))
        url_label_encoder = joblib.load(os.path.join('models', 'url_label_encoder.pkl'))
        return email_model, tfidf_vectorizer, url_model, url_label_encoder
    except FileNotFoundError:
        st.error("Model files not found. Please run 'model_training.py' and 'url_training.py' first.")
        return None, None, None, None

email_model, tfidf_vectorizer, url_model, url_label_encoder = load_models()

# --- Main Streamlit App ---

st.set_page_config(page_title="Email Security Gateway", page_icon="📧", layout="wide")

st.title("📧 Email Security Gateway")
st.markdown("---")
st.markdown("### Paste an email below to check for potential phishing threats.")

email_text = st.text_area("Email Content", height=300, placeholder="Paste the full email here...")

if st.button("Analyze Email", use_container_width=True):
    if not email_text:
        st.warning("Please paste an email to analyze.")
    elif not all([email_model, tfidf_vectorizer, url_model, url_label_encoder]):
        st.error("Models are not loaded. Please ensure all training scripts have been run.")
    else:
        # --- Analysis Logic (same as app.py) ---
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
        
        if email_prediction == 1:
            reasons.append(f"Content analysis model predicts: Phishing (Confidence: {email_prediction_proba[1]:.2f}).")
            final_status = "Phishing Detected"
        else:
            reasons.append(f"Content analysis model predicts: Safe (Confidence: {email_prediction_proba[0]:.2f}).")
        
        # 3. URL Analysis
        urls_found = re.findall(r'https?://\S+', email_text)
        if urls_found:
            reasons.append("---")
            reasons.append("URL Analysis:")
            for url in urls_found:
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
                
                reasons.append(f"- URL: {url} -> Status: {url_status_display}")
        
        # Final decision based on combined analysis
        if url_phishing_count > 0:
            reasons.append("---")
            reasons.append(f"Final Conclusion: {url_phishing_count} suspicious URL(s) detected. ")
            final_status = "Phishing Detected"

        # --- Display Results ---
        st.markdown("---")
        st.markdown("### Analysis Results")
        if final_status == "Phishing Detected":
            st.error("⚠️ Phishing Detected")
        else:
            st.success("✅ Safe Email")

        st.markdown("#### Detailed Breakdown:")
        for reason in reasons:
            st.markdown(f"- {reason}")