import streamlit as st
import pickle
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stopwords.words('english')
ps = PorterStemmer()


def transform_text(text):
    """
    Transform text using the same preprocessing steps as in model training
    """
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y.copy()
    y.clear()
    
    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y.copy()
    y.clear()
    
    # Apply stemming
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf = pickle.load(open('Vectorizer.pkl', 'rb'))
model = pickle.load(open('spam_classifier_model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")


input_sms = st.text_area("Enter the message")   

if st.button("Predict"):

    #1.Preprocess the input text
    transformed_sms = transform_text(input_sms)
    #2.vecorize
    vector_input = tfidf.transform([transformed_sms])
    #3. Predict
    result = model.predict(vector_input)[0]
    #4.Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")