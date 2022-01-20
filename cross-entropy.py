"""
Ever got 😵 confused between CategoricalCrossentropy and SparseCategoricalCrossentropy in 🔥 TensorFlow 🚀

Simply put when true labels are sparse/one-hot encoded, use CategoricalCrossentropy

and

when true labels are non-sparse/integers use SparseCategoricalCrossentropy.

*assuming predicted labels are always one-hot encoded
"""

import numpy as np
import tensorflow as tf

# CategoricalCrossentropy
y_true = [[0, 1, 0], [0, 0, 1]] # one-hot
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

# using TensorFlow
tf_cce = tf.keras.losses.CategoricalCrossentropy()
print("using tf", tf_cce(y_true, y_pred).numpy())

# using numpy
y_pred = np.clip(y_pred, e, 1.-e) # to avoid log(0)
N = len(y_true)  # total number of examples
np_cce = -np.sum(y_true * np.log(y_pred))/N
print("using np", round(np_cce,7))

# using tf 1.1769392
# using np 1.1769392
###############################################

# SparseCategoricalCrossentropy
y_true = [1, 2] # NOT one-hot
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]


# using TensorFlow
tf_cce = tf.keras.losses.SparseCategoricalCrossentropy()
print("using tf", tf_cce(y_true, y_pred).numpy())

# using numpy
y_true = tf.keras.utils.to_categorical([1, 2], num_classes=3, dtype='int')
y_pred = np.clip(y_pred, e, 1.-e) # to avoid log(0)
N = len(y_true)  # total number of examples
np_cce = -np.sum(y_true * np.log(y_pred))/N
print("using np", round(np_cce,7))

# using tf 1.1769392
# using np 1.1769392
