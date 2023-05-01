import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import os

# Initialize empty lists for text and title
text_list = []
title_list = []

# Traverse the Krapivin2009 dataset folders and read the .txt files
for root, dirs, files in os.walk('Krapivin2009'):
    for file in files:
        if file.endswith('.txt'):
            # Read the text and use the file name as the title
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
            title = file.replace('.txt', '')
            
            # Add the text and title to the lists
            text_list.append(text)
            title_list.append(title)

# Create a dataframe from the lists
df = pd.DataFrame({'text': text_list, 'title': title_list})

# Tokenize the words in the corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'].values.tolist())
total_words = len(tokenizer.word_index) + 1

# Create sequences of tokens
input_sequences = []
for line in df['text']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to be of equal length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and label
X, y = input_sequences[:,:-1],input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the model
history = model.fit(X, y, epochs=50, verbose=1)

# Generate title for given text
def generate_title(text):
    text_sequence = tokenizer.texts_to_sequences([text])[0]
    text_sequence = pad_sequences([text_sequence], maxlen=max_sequence_len-1, padding='pre')
    prediction = model.predict(text_sequence)
    predicted_word_index = np.argmax(prediction)
    predicted_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            predicted_word = word
            break
    return predicted_word

# Example usage
generated_title = generate_title("This is a sample text for title generation using LSTM")
print(generated_title)
