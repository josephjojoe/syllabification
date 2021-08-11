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
