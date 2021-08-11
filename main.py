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

# Custom loss function - mean of binary crossentropy and mean squared error
def mean_weighted_bce_mse(y_true, y_prediction):
    y_true = tf.cast(y_true, tf.float32)
    y_prediction = tf.cast(y_prediction, tf.float32)

    # Binary crossentropy with weighting
    epsilon = 1e-6
    positive_weight = 4.108897148948174
    loss_positive = y_true * tf.math.log(y_prediction + epsilon)
    loss_negative = (1 - y_true) * tf.math.log(1 - y_prediction + epsilon)
    bce_loss = tf.math.reduce_mean(tf.math.negative(positive_weight * loss_positive + loss_negative))
    
    # Mean squared error
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(y_true, y_prediction)

    averaged_bce_mse = (bce_loss + mse_loss) / 2
    return averaged_bce_mse
