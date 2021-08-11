import tensorflow as tf
import tensorflow_addons as tfa
from loss import mean_weighted_bce_mse


def build_model():
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

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    metrics = ["binary_accuracy",
               tfa.metrics.F1Score(num_classes=15, threshold=0.5),
               tfa.metrics.HammingLoss(mode='multilabel', threshold=0.5),
               tf.keras.metrics.Recall(),
               tf.keras.metrics.Precision(),
               tf.keras.metrics.AUC(multi_label=True, num_labels=15)]

    model.compile(optimizer="adam",
                  loss=mean_weighted_bce_mse,
                  metrics=metrics,
                  steps_per_execution=64)

    try:
        model.load_weights("./my_model/ckpt")
        print("Loading of model weights successful.")
    except:
        print("No model weights found.")
        pass
    
    return model
