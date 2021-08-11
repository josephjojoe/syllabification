# Imports dependencies
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import random


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
    
# Pads inputs/outputs to desired maximum sequence length
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    words, padding="post", maxlen=15
)
padded_outputs = tf.keras.preprocessing.sequence.pad_sequences(
    labels, padding="post", maxlen=15
)    
