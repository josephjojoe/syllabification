import tensorflow as tf
import tensorflow_addons as tfa
from loss import mean_weighted_bce_mse


def build_model():
    inputs = tf.keras.Input(shape=(15,))
    embedded_inputs = tf.keras.layers.Embedding(64, 128, mask_zero=True)(inputs)
    embedded_inputs = tf.keras.layers.BatchNormalization()(embedded_inputs)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1, activity_regularizer=tf.keras.regularizers.l2(1e-5)))(embedded_inputs)
    x = tf.keras.layers.concatenate([x, embedded_inputs])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1, activity_regularizer=tf.keras.regularizers.l2(1e-5)))(x)
    x = tf.keras.layers.concatenate([x, embedded_inputs])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, activity_regularizer=tf.keras.regularizers.l2(1e-5)))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    inception_output = inception_module(embedded_inputs)

    output = tf.keras.layers.concatenate([inception_output, embedded_inputs])
    output = tf.keras.layers.add([output, x])
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation="relu"))(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation="relu"))(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.GlobalMaxPool1D()(output)
    output = tf.keras.layers.Dense(15, activation="sigmoid")(output)

    metrics = ["binary_accuracy",
               tfa.metrics.F1Score(num_classes=15, threshold=0.5),
               tfa.metrics.HammingLoss(mode='multilabel', threshold=0.5),
               tf.keras.metrics.Recall(),
               tf.keras.metrics.Precision(),
               tf.keras.metrics.AUC(multi_label=True, num_labels=15)]

    model = tf.keras.models.Model(inputs=inputs, outputs=output)
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
