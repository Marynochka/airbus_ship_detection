import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice coefficient between the true and predicted values.

    Args:
        y_true (list): List of the true values for each image.
        y_pred (list): List of the predicted values for each image.
        smooth (float, optional): Smoothing factor to avoid division by zero, 1e-6 by default.

    Returns:
        tf.Tensor: The Dice coefficient between y_true and y_pred.
    """
    y_true_d = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_d = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_d * y_pred_d)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_d) + tf.reduce_sum(y_pred_d) + smooth)
    return dice
def dice_loss(y_true, y_pred):
    """
    Computes the Dice loss between the true and predicted values.

    Args:
        y_true (list): List of the true values for each image.
        y_pred (list): List of the predicted values for each image.

    Returns:
        tf.Tensor: The Dice loss between y_true and y_pred.
    """
    return -dice_coef(y_true, y_pred)
