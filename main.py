# Imports dependencies
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import random
from dataloader import load_data
from model import build_model

# Loads data
train_in, train_out, val_in, val_out, test_in, test_out = load_data()

# Builds model and loads weights if present
model = build_model()

# Callbacks
callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy',
                                            mode='max', patience=10,
                                            restore_best_weights=True)

# Fits model and saves weights at the end
history = model.fit(train_in,
                    train_out,
                    validation_data=(val_in, val_out),
                    epochs=10,
                    batch_size=8,
                    callbacks=[callback],
                    verbose=1)
model.save_weights("./my_model/ckpt")
