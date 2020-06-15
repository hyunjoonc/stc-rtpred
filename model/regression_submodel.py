import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

_input_columns = [
    'par.procs', 'sz.data'
]

_output_columns = [
    'res.time'
]

def model(weights = None):
    inputs = keras.layers.Input(shape = (2,)) # np,sz
    
    y = keras.layers.Dense(5, activation = tf.nn.elu)(inputs)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Dense(1)(y)
    
    m = keras.Model(inputs = inputs, outputs = y)
    
    if weights is not None:
        m.load_weights(weights)
    
    return m

