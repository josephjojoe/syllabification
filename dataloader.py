import tensorflow as tf
import pandas as pd
import numpy as np
import random

def load_data():
    # Loads in syllable data
    dataframe = pd.read_csv("./dataset/preprocessed.txt",
                            sep=",",
                            encoding="ISO-8859-1",
                            names=["word", "label"])
    # Necessary to specify str type for pandas columns
    dataframe = dataframe.astype(str)
    words = dataframe['word'].tolist()
    labels = dataframe['label'].tolist()
    # Converts each label to numpy array
    for i in range(0, len(labels)):
        labels[i] = list(labels[i])
        for j in range(0, len(labels[i])):
            labels[i][j] = int(labels[i][j])
    for i in range(0, len(labels)):
        labels[i] = np.array(labels[i])

    # Vectorises syllable strings by treating each character as a token
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(words)
    words = tokenizer.texts_to_sequences(words)
    for i in range(0, len(words)):
        words[i] = np.array(words[i], dtype=float)

    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        words, padding="post", maxlen=15
    )
    padded_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        labels, padding="post", maxlen=15
    )

    # Normalisation
    maximum_token = 37
    for element in range(0, len(words)):
        words[element] = words[element] / maximum_token

    # Shuffles data
    seed = random.random()
    random.seed(seed)
    random.shuffle(padded_inputs)
    random.seed(seed)
    random.shuffle(padded_outputs)

    # Splits into training, validation, and test sets (64-16-20 split)
    training_inputs = padded_inputs[0:113590]
    training_outputs = padded_outputs[0:113590]
    validation_inputs = padded_inputs[113590:141987]
    validation_outputs = padded_outputs[113590:141987]
    test_inputs = padded_inputs[141987:]
    test_outputs = padded_outputs[141987:]

    return training_inputs, training_outputs, validation_inputs, validation_outputs, test_inputs, test_outputs
