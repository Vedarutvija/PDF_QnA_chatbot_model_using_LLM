from PyPDF2 import PdfReader
import streamlit as st
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import numpy as np

# Load the PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
document = ''

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        document += page.extract_text()

# Tokenize and preprocess the document
tokenizer = Tokenizer()
tokenizer.fit_on_texts([document])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in document.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Check if input_sequences is empty
if not input_sequences:
    st.error("Error processing the document. Please try again with a different PDF.")
else:
    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    # Build a simple neural network model
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_length - 1))
    model.add(Flatten())
    model.add(Dense(total_words, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=1)

    # Get answers for the question using the trained model
    question = st.text_input("Ask a question:")
    if question and uploaded_file:
        input_text = [document + ' ' + question]
        input_seq = tokenizer.texts_to_sequences(input_text)
        input_seq = pad_sequences(input_seq, maxlen=max_sequence_length - 1, padding='pre')
        predicted_word_index = np.argmax(model.predict(input_seq), axis=-1)
        predicted_word = tokenizer.index_word[predicted_word_index[0]]

        st.success("Top Answer:")
        st.write(predicted_word)
