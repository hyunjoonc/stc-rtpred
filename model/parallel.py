import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

_input_columns = [
    'par.procs', 'par.hosts'
]

_output_columns = [
    'res.time'
]

def model(x, output_dim = 2, weights = None):
    y = keras.layers.Dense(8, kernel_regularizer = keras.regularizers.l2(0.004), activation = tf.nn.leaky_relu)(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(8, kernel_regularizer = keras.regularizers.l2(0.004), activation = tf.nn.leaky_relu)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(8, kernel_regularizer = keras.regularizers.l2(0.004), activation = tf.nn.leaky_relu)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(8, kernel_regularizer = keras.regularizers.l2(0.004), activation = tf.nn.leaky_relu)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(8, kernel_regularizer = keras.regularizers.l2(0.004), activation = tf.nn.leaky_relu)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(output_dim)(y)
    
    return y
