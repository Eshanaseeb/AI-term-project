import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load your trained model and tokenizer
@st.cache_resource
def load_resources():
    model = load_model('LSTM_model.h5')  # Replace with the path to your saved model
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)  # Replace with the path to your tokenizer
    return model, tokenizer

model, tokenizer = load_resources()

# Define class labels
class_labels = ['Neutral', 'Negative', 'irrelevent', 'positive']

# Streamlit app
st.title("Tweet Sentiment Analysis")
st.write("This app predicts the sentiment of a given tweet.")

# User input
tweet = st.text_area("Enter a Tweet:", "")

# Predict sentiment
if st.button("Predict Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a valid tweet.")
    else:
        # Preprocess the tweet
        max_len = 100  # Update this based on the max length used during training
        sequence = tokenizer.texts_to_sequences([tweet])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

        # Predict sentiment
        predictions = model.predict(padded_sequence)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        # Display results
        st.subheader("Prediction Results:")
        st.write(f"**Tweet:** {tweet}")
        st.write(f"**Predicted Sentiment:** {class_labels[predicted_class]}")
        # Optional: Display probabilities for each class
        st.write("**Class Probabilities:**")
        for i, label in enumerate(class_labels):
            st.write(f"{label}: {predictions[0][i]:.2f}")
