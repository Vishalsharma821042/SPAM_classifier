import streamlit as st
import pickle
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))  # Cache stopwords once

def transform_text(text):
    """
    Transform text using the same preprocessing steps as in model training
    """
    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    tokens = [i for i in tokens if i.isalnum()]

    # Remove stopwords and punctuation
    tokens = [i for i in tokens if i not in stop_words and i not in string.punctuation]

    # Stemming
    tokens = [ps.stem(i) for i in tokens]

    return " ".join(tokens)

# Load vectorizer and model
tfidf = pickle.load(open('Vectorizer.pkl', 'rb'))
model = pickle.load(open('spam_classifier_model.pkl', 'rb'))

# UI
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Output
    st.header("Spam" if result == 1 else "Not Spam")
