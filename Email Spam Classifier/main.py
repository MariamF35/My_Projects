"""
Email Spam Classifier
====================
This Python program uses scikit-learn to classify emails/messages as SPAM or HAM.
It demonstrates the following concepts:
- reading CSV files with pandas
- basic text preprocessing and vectorization
- training and evaluating a Naive Bayes classifier
"""
# Install required libraries 
# !pip install scikit-learn pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from google.colab import files
# -----------------------------
# Upload Dataset (spam.csv)
# -----------------------------
uploaded = files.upload()  # GUI to upload your CSV file

# Read the uploaded CSV
df = pd.read_csv(next(iter(uploaded.keys())), encoding='latin-1')

# Keep only necessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Map labels to numerical values
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# -----------------------------
# Split dataset into train/test
# -----------------------------
X = df['message']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Text vectorization
# -----------------------------
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Train Naive Bayes classifier
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Interactive testing loop
# -----------------------------
print("\nTest your own messages! Type 'exit' to quit.")
while True:
    msg = input("Enter a message: ")
    if msg.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        break
    elif msg.strip() == '':
        # Skip empty messages
        print("⚠️ Please enter a valid message.")
        pass
    else:
        msg_vec = vectorizer.transform([msg])
        prediction = model.predict(msg_vec)
        if prediction[0] == 1:
            print("⚠️ This message is likely SPAM!")
        else:
            print("✅ This message is HAM (not spam).")
