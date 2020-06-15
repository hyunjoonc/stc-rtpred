import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools, functools

import math
import time
import importlib
import os, sys
from tqdm import tqdm

import model.model as model

# data setup. stc.mpic column is the state complexity of mpi collective (gather, scatter) routines.
input_columns = ['stc.code', 'stc.mpi', 'stc.mpic', 'problem.n', 'problem.nb', 'distr.procs', 'distr.hosts']
output_columns = ['result.time']

def data_preproc(df):
    for key in ['problem.n', 'problem.nb', 'stc.code', 'stc.mpi', 'stc.mpic']:
        if key in df:
            df[key] = np.log(df[key] + 0.001)
    
    return df.sample(frac = 1.0)

data = pd.read_csv('data/data.csv')
data = data[ data['meta.config'] == 'sw2' ][:1178]
data = data_preproc(data)
input_data = data[input_columns]
output_data = data[output_columns]

dataset = {}

pivots = {
    'train': (0, int(len(data) * 0.7)),
    'dev': (int(len(data) * 0.7), int(len(data) * 0.9)),
    'test': (int(len(data) * 0.9), len(data))
}

for k, v in pivots.items():
    dataset[k] = tf.data.Dataset.from_tensor_slices(
        ( input_data.values[pivots[k][0]:pivots[k][1]], output_data.values[pivots[k][0]:pivots[k][1]] )
    )
    
    dataset[k] = dataset[k].shuffle(500).batch(100)

m = model.model()

lr_fn = tf.optimizers.schedules.PolynomialDecay(0.4, 2000, 0.001, power = 0.5)
optimizer = tf.optimizers.Adadelta(lr_fn)
loss_fn = keras.losses.MeanSquaredError()

train_loss = keras.metrics.Mean()
test_loss = keras.metrics.Mean()

@tf.function
def train_step(model, inputs, outputs):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training = True)
        loss = loss_fn(outputs, predictions) + sum(model.losses)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss
    
@tf.function
def test_step(model, inputs, outputs):
    predictions = model(inputs, training = False)
    loss = loss_fn(outputs, predictions) + sum(model.losses)
    
    return loss

val_loss = 150
        
progress = tqdm(range(5000), ncols=78, ascii=True, desc='Training')
for epoch in progress:
    train_loss.reset_states()
    test_loss.reset_states()
    
    for inputs, outputs in dataset['train']:
        l = train_step(m, inputs, outputs)
        train_loss.update_state(l)
    
    for inputs, outputs in dataset['test']:
        l = test_step(m, inputs, outputs)
        test_loss.update_state(l)
    
    progress.set_description('#{:05d} {:.2f}~{:.2f} (lr={:.3f})'.format(epoch, train_loss.result(), test_loss.result(), lr_fn(epoch)))

    if test_loss.result() < val_loss:
        val_loss = test_loss.result()

m.save_weights(model.MODEL_WEIGHTS)