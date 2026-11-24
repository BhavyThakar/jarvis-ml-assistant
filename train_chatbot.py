import numpy as np
import pickle

# --------------------------
# Keras 3.x + TF 2.16 imports
# --------------------------
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from keras.optimizers import Adam

# Tokenizer + pad_sequences now come from keras-preprocessing
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# to_categorical still comes from tf.keras
from tensorflow.keras.utils import to_categorical

# sklearn utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


print("\n🚀 Training starting...\n")

# -----------------------------------
# LOAD DATA
# -----------------------------------
input_texts = []
output_texts = []

with open("training_data.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "=" not in line:
            continue
        q, a = line.split("=", 1)
        q = q.strip().lower()
        a = a.strip()
        if q and a:
            input_texts.append(q)
            output_texts.append(a)

print("📌 Total pairs loaded:", len(input_texts))


# -----------------------------------
# TOKENIZER SETUP
# -----------------------------------
tokenizer_inp = Tokenizer(oov_token="<OOV>")
tokenizer_inp.fit_on_texts(input_texts)

sequences = tokenizer_inp.texts_to_sequences(input_texts)
max_len = max(len(s) for s in sequences)

X = pad_sequences(sequences, maxlen=max_len, padding="post")

vocab_size = len(tokenizer_inp.word_index) + 1

print("📌 Vocab size:", vocab_size)
print("📌 Max input length:", max_len)


# -----------------------------------
# LABEL ENCODER
# -----------------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(output_texts)
num_classes = len(label_encoder.classes_)

y_cat = to_categorical(y, num_classes=num_classes)

print("📌 Number of unique responses:", num_classes)


# -----------------------------------
# TRAIN / VALIDATION SPLIT
# -----------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.15, random_state=42
)

print("\nDataset split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# -----------------------------------
# MODEL ARCHITECTURE
# -----------------------------------
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.003),
    metrics=["accuracy"]
)

print("\n🧠 Model Summary:")
model.summary()


# -----------------------------------
# TRAIN MODEL
# -----------------------------------
print("\n📚 Training model...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    verbose=1
)

print("\n🎉 Training complete!")


# -----------------------------------
# SAVE OUTPUT FILES
# -----------------------------------
model.save("chatbot_model.h5")
print("💾 Saved: chatbot_model.h5")

with open("tokenizer_inp.pkl", "wb") as f:
    pickle.dump(tokenizer_inp, f)
print("💾 Saved: tokenizer_inp.pkl")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("💾 Saved: label_encoder.pkl")

print("\n✅ ALL DONE! Your model is ready to use.\n")
