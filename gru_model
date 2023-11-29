import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pdfplumber
import numpy as np
import nltk
from keras.layers import GRU, Bidirectional, RepeatVector, TimeDistributed, Dense, Embedding
from keras.models import Sequential

nltk.download('punkt')
nltk.download('stopwords')

file_path = "/kaggle/input/question/story.pdf"
document = ''

pdf_reader = pdfplumber.open(file_path)
for page in pdf_reader.pages:
    document += page.extract_text()

# Tokenize and preprocess the document
tokenizer = Tokenizer()
tokenizer.fit_on_texts([document])
total_words = len(tokenizer.word_index) + 1

token_list = tokenizer.texts_to_sequences([document])[0]
input_sequences = []
for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i + 1]
    input_sequences.append(n_gram_sequence)

else:
    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    X, y = input_sequences[:, :-1], input_sequences[:, 1:]  # Modified this line
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    
    # Use an encoder-decoder model with GRU layers
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_length - 1))
    model.add(Bidirectional(GRU(200, return_sequences=True)))  # Changed LSTM to GRU
    model.add(GRU(200, return_sequences=True))  # Changed LSTM to GRU
    model.add(TimeDistributed(Dense(total_words, activation='softmax')))

    # Train the model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=1, callbacks=[early_stopping])
    model.summary()

    # Generate relevant answers to questions
    def generate_answer(question, document, model, tokenizer, max_sequence_length):
        input_text = [document + ' ' + question]
        input_seq = tokenizer.texts_to_sequences(input_text)
        input_seq = pad_sequences(input_seq, maxlen=max_sequence_length - 1, padding='pre')
        predicted_sequence = model.predict(input_seq)
        predicted_sequence = np.argmax(predicted_sequence, axis=-1)[0]

        # Filter out-of-vocabulary indices
        predicted_words = [tokenizer.index_word[word_index] for word_index in predicted_sequence if word_index in tokenizer.index_word]

        return ' '.join(predicted_words)


    # Test with a question
    question = "What is the name of Veena's sibling?"
    answer = generate_answer(question, document, model, tokenizer, max_sequence_length)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
