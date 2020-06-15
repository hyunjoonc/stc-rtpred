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
import random
from tqdm import tqdm

import model.regression_submodel as rs

#importlib.reload()

# tensorflow gpu setup
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 512)])
    except RuntimeError:
        pass

def get_data(name):
    df = pd.read_csv(name)
    return df

data = get_data('data/osu.csv')

types = [
    'allgather', 'allgatherv',
    'allreduce', 
    'alltoall', 'alltoallv',
    'bcast',
    'gather', 'gatherv',
    'reduce',
    'reduce_scatter',
    'scatter',
    'scatterv']

data['time'] = np.log(data['time'])
data['sz'] = np.log(data['sz'])

ix = list(range(0, len(data[ data['type'] == types[0] ])))
random.shuffle(ix)

idx = {
    'train': ix[:22],
    'dev': ix[22:]
}

model = {}
for k in types:
    model[k] = rs.model()

dataset = {}

for t in types:
    dataset[t] = {}
    for k, ix in idx.items():
        d = data[ data['type'] == t ]
        dataset[t][k] = tf.data.Dataset.from_tensor_slices(
            ( d[['np', 'sz']].values[ix].astype(np.float64), d[ ['time'] ].values[ix] )
        ).batch(500)

lr_fn = tf.optimizers.schedules.PolynomialDecay(0.1, 10000, 5e-3, power = 0.8)
optimizer = tf.optimizers.Adadelta(lr_fn)
loss_fn = keras.losses.MeanSquaredError()

train_loss = keras.metrics.Mean()
test_loss = keras.metrics.Mean()

for t in types:
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

    val_loss = 0.05

    progress = tqdm(range(10000), ncols=120, ascii=True, desc=t)
    for epoch in progress:
        train_loss.reset_states()
        test_loss.reset_states()
        
        for inputs, outputs in dataset[t]['train']:
            l = train_step(model[t], inputs, outputs)
            train_loss.update_state(l)
        for inputs, outputs in dataset[t]['dev']:
            test_step(model[t], inputs, outputs)
            test_loss.update_state(l)

        progress.set_description('#{:05d} {:.3f}~{:.3f} (lr={:.4f})'.format(epoch, train_loss.result(), test_loss.result(), lr_fn(epoch)))
        
        if test_loss.result() < 0.01:
            break

    model[t].save_weights('weights/mpimodel_{}.ckpt'.format(t))
