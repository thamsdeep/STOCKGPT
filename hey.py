import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = tf.keras.models.load_model('path_to_your_model/model.h5')
with open('path_to_your_tokenizer/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
max_len = 500

st.title("Stock Market News Sentiment Analyzer")

input_headline = st.text_area("Enter a news headline:")

if st.button("Predict"):
    if input_headline.strip():

        sequence = tokenizer.texts_to_sequences([input_headline])
        padded_sequence = pad_sequences(sequence, maxlen=max_len)

        # Get prediction
        prediction = model.predict(padded_sequence)[0][0]

        if prediction > 0.5:
            st.success("The news is likely to have a **Good Effect** on the stock market.")
        else:
            st.error("The news is likely to have a **Bad Effect** on the stock market.")
    else:
        st.warning("Please enter a headline to analyze.")

# Option to display model's prediction confidence
if st.checkbox("Show prediction confidence"):
    if input_headline.strip():
        st.write(f"Prediction confidence: {prediction:.4f}")
