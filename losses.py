import tensorflow.keras.backend as K
import tensorflow as tf


def dsc(y_true, y_pred):
    smooth = 1.0

    # Flatten and cast to float32
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))  # Ensure y_true is float32
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))  # Ensure y_pred is float32
    
    # Compute intersection and Dice coefficient
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    return dice

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def weighted_dice_loss(y_true, y_pred, weight=2.0):
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(weight * y_true * y_pred)
    denominator = tf.reduce_sum(weight * y_true + y_pred)
    return 1 - (2 * intersection + tf.keras.backend.epsilon()) / (denominator + tf.keras.backend.epsilon())

def weighted_bce(y_true, y_pred):
    weights = (y_true * 4.) + 1.
    weights = tf.squeeze(weights, axis=-1)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weighted_bce = tf.reduce_mean(weights * bce)
    return weighted_bce