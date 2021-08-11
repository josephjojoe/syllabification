import tensorflow as tf

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
