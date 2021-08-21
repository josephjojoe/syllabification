import tensorflow as tf
import tensorflow_addons as tfa
from loss import mean_weighted_bce_mse


def build_model():
    # Builds syllabification model with Keras Functional API.

    def inception_module(inputs):
        # 1D version of Inception module, with residual connections.
        inception_branch_1 = tf.keras.layers.Conv1D(32, kernel_size=1, strides=2, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(1e-5))(inputs)
        inception_branch_1 = tf.keras.layers.ZeroPadding1D(padding=(0, 15 - inception_branch_1.shape[1]))(inception_branch_1)

        inception_branch_2 = tf.keras.layers.Conv1D(32, kernel_size=1, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(1e-5))(inputs)
        inception_branch_2 = tf.keras.layers.Conv1D(32, kernel_size=3, strides=2, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(1e-5))(inception_branch_2)
        inception_branch_2 = tf.keras.layers.ZeroPadding1D(padding=(0, 15 - inception_branch_2.shape[1]))(inception_branch_2)

        inception_branch_3 = tf.keras.layers.AveragePooling1D(pool_size=3, strides=2)(inputs)
        inception_branch_3 = tf.keras.layers.Conv1D(32, kernel_size=3, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(1e-5))(inception_branch_3)
        inception_branch_3 = tf.keras.layers.ZeroPadding1D(padding=(0, 15 - inception_branch_3.shape[1]))(inception_branch_3)

        inception_branch_4 = tf.keras.layers.Conv1D(32, kernel_size=1, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(1e-5))(inputs)
        inception_branch_4 = tf.keras.layers.Conv1D(32, kernel_size=3, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(1e-5))(inception_branch_4)
        inception_branch_4 = tf.keras.layers.Conv1D(32, kernel_size=3, strides=2, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(1e-5))(inception_branch_4)
        inception_branch_4 = tf.keras.layers.ZeroPadding1D(padding=(0, 15 - inception_branch_4.shape[1]))(inception_branch_4)

        inception_output = tf.keras.layers.concatenate([inception_branch_1, inception_branch_2, inception_branch_3, inception_branch_4, inputs])
        return inception_output

    # Main model building code.
    inputs = tf.keras.Input(shape=(15,))
    embedded_inputs = tf.keras.layers.Embedding(64, 64, mask_zero=True)(inputs)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1, activity_regularizer=tf.keras.regularizers.l2(1e-5)))(embedded_inputs)
    x = tf.keras.layers.concatenate([x, embedded_inputs])
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1, activity_regularizer=tf.keras.regularizers.l2(1e-5)))(x)
    x = tf.keras.layers.concatenate([x, embedded_inputs])
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, activity_regularizer=tf.keras.regularizers.l2(1e-5)))(x)

    inception_output = inception_module(embedded_inputs)

    output = tf.keras.layers.concatenate([x, inception_output, embedded_inputs])
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation="relu"))(output)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation="relu"))(output)
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
