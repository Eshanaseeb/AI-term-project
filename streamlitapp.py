import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
@st.cache_resource
def load_resources():
    model = load_model('LSTM_model.h5')  # Ensure this file exists
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# App title
st.title("Twitter Sentiment Analysis")

# Sidebar information
st.sidebar.title("About")
st.sidebar.write("""
This app analyzes the sentiment of tweets using an LSTM model trained on Twitter data.
""")

# Input text from the user
st.write("Enter a tweet to analyze its sentiment:")
user_input = st.text_area("Tweet", placeholder="Type your tweet here...")

# Preprocessing function
def preprocess_text(text):
    import re
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)    # Remove mentions
    text = re.sub(r'#\w+', '', text)    # Remove hashtags
    text = re.sub(r'\d+', '', text)     # Remove numbers
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text.lower().strip()

# Predict function
def predict_sentiment(text, model, tokenizer, max_length=100):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    sentiment_classes = ['Neutral', 'Negative', 'Irrelevent', 'Positive']
    sentiment = sentiment_classes[np.argmax(prediction)]
    return sentiment, prediction

# Predict sentiment on button click
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment, probabilities = predict_sentiment(user_input, model, tokenizer)
        st.write(f"**Sentiment:** {sentiment}")
        st.write("**Probabilities:**")
        st.write(pd.DataFrame(
            data=[probabilities[0]],
            columns=['Positive', 'Neutral', 'Negative', 'Irrelevant']
        ))
    else:
        st.error("Please enter a tweet to analyze.")


