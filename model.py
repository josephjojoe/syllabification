# Imports dependencies
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import random
from loss import mean_weighted_bce_mse


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

# Builds syllabification model with Keras Functional API.
inputs = tf.keras.Input(shape=(15,))
embedded_inputs = tf.keras.layers.Embedding(64, 64, mask_zero=True)(inputs)
 
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1, activity_regularizer=tf.keras.regularizers.l2(1e-5)))(embedded_inputs)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1, activity_regularizer=tf.keras.regularizers.l2(1e-5)))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, activity_regularizer=tf.keras.regularizers.l2(1e-5)))(x)

y = tf.keras.layers.Conv1D(128, kernel_size=1, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(1e-5))(embedded_inputs)
y = tf.keras.layers.Dropout(0.5)(y)
y = tf.keras.layers.Conv1D(128, kernel_size=1, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(1e-5))(y)
y = tf.keras.layers.Dropout(0.5)(y)
y = tf.keras.layers.Conv1D(128, kernel_size=1, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(1e-5))(y)

merged_outputs = tf.keras.layers.concatenate([x, y])

x = tf.keras.layers.Dropout(0.5)(merged_outputs)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="relu"))(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation="relu"))(x)
x = tf.keras.layers.GlobalMaxPool1D()(x)
x = tf.keras.layers.Dense(15, activation="sigmoid")(x)

metrics = ["binary_accuracy",
           tfa.metrics.F1Score(num_classes=15, threshold=0.5),
           tfa.metrics.HammingLoss(mode='multilabel', threshold=0.5),
           tf.keras.metrics.Recall(),
           tf.keras.metrics.Precision(),
           tf.keras.metrics.AUC(multi_label=True, num_labels=15)]

model = tf.keras.models.Model(inputs=inputs, outputs=x)
model.compile(optimizer="adam",
              loss=mean_weighted_bce_mse,
              metrics=metrics,
              steps_per_execution=64)

model.fit(training_inputs,
          training_outputs,
          validation_data=(validation_inputs, validation_outputs),
          epochs=5,
          batch_size=8,
          verbose=1)
