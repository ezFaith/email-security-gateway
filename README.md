Email Security Gateway with Phishing DetectionOverviewThis is a complete Python-based project that serves as an email security gateway. It is designed to detect and flag potential phishing emails by performing a multi-layered analysis of email content and URLs.The core of the system is a machine learning model trained on a large dataset of legitimate and spam emails. This is complemented by a rule-based system for advanced URL analysis, which makes the detector more robust and accurate.Key FeaturesContent Analysis: Utilizes a Random Forest Classifier trained on a large email corpus to predict whether an email is spam/phishing based on its text.URL Analysis: Extracts all URLs from an email and performs a separate analysis using a dedicated URL classification model.Rule-Based Detection: Incorporates a simple but effective rule-based check for suspicious Top-Level Domains (TLDs) like .xyz and .info, which are commonly used in phishing campaigns.Web Interface: A simple and clean web interface built with Flask allows users to paste an email and receive an immediate analysis report.Clear Reporting: Provides a detailed breakdown of the analysis, including confidence scores, keyword matches, and the status of each URL found.Tools UsedPython: The core programming language for the project.Flask: A micro-framework for building the web application.scikit-learn: A powerful machine learning library used for training the classification models.Pandas: Used for efficient data handling and preprocessing.Joblib: For saving and loading the trained models.urlextract: A library for reliably extracting URLs from text.Project Structureemail-security-gateway/

├── app.py                  # The Flask web application.

├── models/

│   ├── phishing\_model.pkl    # The trained email classification model.

│   ├── tfidf\_vectorizer.pkl  # The TF-IDF vectorizer for email text.

│   ├── url\_model.pkl         # The trained URL classification model.

│   └── url\_label\_encoder.pkl # Label encoder for URL predictions.

├── model\_training.py       # Script for training the email model.

├── url\_training.py         # Script for training the URL model.

├── requirements.txt        # List of Python dependencies.

├── static/

│   └── style.css           # (Optional) For custom styling.

└── templates/

    └── index.html          # The HTML template for the web interface.

Setup InstructionsClone the Repository:git clone \[your-repo-url]

cd email-security-gateway

Install Dependencies:Make sure you have Python installed. Then, run the following command to install the required libraries:pip install -r requirements.txt

Download Datasets:You will need two main datasets from Kaggle:Spam and Ham Email Dataset: A large dataset of emails for the content model.Download emails.csv and place it in the root directory.Phishing URLs Dataset: A dataset of URLs for the URL model.Download dataset\_phishing.csv and place it in the root directory.Train the Models:First, train the email model:python model\_training.py

Next, train the URL model:python url\_training.py

This will create the necessary .pkl files in the models/ directory.How to Run the ApplicationOnce the models are trained, start the Flask web server:python app.py

The server will start running on your local machine. Open your web browser and navigate to http://127.0.0.1:5000/.Example Test Cases You can test the application using these sample emails:

Phishing Example:


Subject: Security Alert: We detected an unrecognized sign-in

Dear Customer,



We have detected unusual activity on your account. A sign-in was made from an unfamiliar device. To protect your account, we have temporarily suspended your access.



Please click the link below to verify your identity and remove the security suspension.



\[https://login-my-account-secure.info/verification/page](https://login-my-account-secure.info/verification/page)



Failure to do so will result in permanent account deactivation.



Thank you,

The Security Team


Legitimate Example:


Subject: Project Update

Hi team,



Just a quick update on the project. The latest report is attached. Let's touch base tomorrow morning to confirm the next steps.



Best,

Sarah

The project is now fully documented and ready for use. If you have any further questions or would like to add more features, feel free to ask!

