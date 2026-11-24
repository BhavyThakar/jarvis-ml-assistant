import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import json

# Step 1: Load training data from file
training_data = {}
with open("training_data.txt", "r", encoding="utf-8") as file:
    for line in file:
        if '=' in line:
            question, answer = line.strip().split("=", 1)
            training_data[question.strip()] = answer.strip()

# Step 2: Prepare sequences
questions = list(training_data.keys())
answers = list(training_data.values())

# Input tokenizer
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(questions)
input_sequences = input_tokenizer.texts_to_sequences(questions)
input_padded = pad_sequences(input_sequences, padding='post')

# Output tokenizer
output_tokenizer = Tokenizer()
output_tokenizer.fit_on_texts(answers)
output_sequences = output_tokenizer.texts_to_sequences(answers)
output_padded = pad_sequences(output_sequences, padding='post')

output_vocab_size = len(output_tokenizer.word_index) + 1

# Convert output to categorical (one-hot vectors)
output_categorical = to_categorical(output_padded, num_classes=output_vocab_size)

# Step 3: Build model
model = Sequential()
model.add(Embedding(input_dim=len(input_tokenizer.word_index) + 1, output_dim=64))
model.add(LSTM(64))
model.add(Dense(output_vocab_size, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model: use only first word of output sequences for classification
model.fit(input_padded, output_categorical[:, 0, :], epochs=100, verbose=1)

# Save model and tokenizers
model.save("chat_model.h5")

with open("input_tokenizer.json", "w") as f:
    f.write(input_tokenizer.to_json())

with open("output_tokenizer.json", "w") as f:
    f.write(output_tokenizer.to_json())