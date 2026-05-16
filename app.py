import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

# Load model
model = pickle.load(open("model.pkl","rb"))

vectorizer = pickle.load(
    open("vectorizer.pkl","rb")
)

stop_words = set(stopwords.words("english"))

# Clean text
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Prediction
def predict_job(post):

    cleaned = clean_text(post)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    probability = model.predict_proba(vector)

    fake_prob = probability[0][1]

    if prediction[0] == 1:

        return f"⚠️ Fake Job Posting ({fake_prob:.2f})"

    else:

        return f"✅ Real Job Posting ({1-fake_prob:.2f})"

# UI

st.title("Fake Job Posting Detection System")

st.write(
"Detect fraudulent job postings using NLP and Machine Learning."
)

job_text = st.text_area(
"Enter Job Description"
)

if st.button("Analyze Job Posting"):

    result = predict_job(job_text)

    st.success(result)