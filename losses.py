import tensorflow.keras.backend as K
import tensorflow as tf

smooth = 1

def dsc(y_true, y_pred):
    smooth = 1.
    # Ensure data types match
    y_true_f = K.flatten(tf.cast(y_true, tf.float32)) # Cast y_true to float32
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss