import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np

from . import refine_stc as refstc
from . import parallel as par
from . import regression_submodel as rs

tfd = tfp.distributions

_input_columns = [
    'stc.code', 'stc.mpi', 'stc.mpic',
    'sz.prob', 'sz.subprob',
    'par.procs', 'par.hosts'
]

_output_columns = [
    'res.time'
]

MODEL_WEIGHTS = 'weights/weights.ckpt'
MODEL_INC_WEIGHTS = 'weights/incweights.ckpt'

def get_submodel(x, dim = 2):
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
    y = keras.layers.Dense(dim)(y)
    return y

def model():
    inputs = keras.layers.Input(shape = (len(_input_columns),))
    
    # ['stc.code', 'stc.mpi', 'mpic', problem.n', 'problem.nb', 'distr.procs', 'distr.hosts']
    sc, sm, smc, pn, pb, dp, dh = keras.layers.Lambda( lambda x: tf.split(x, axis = 1, num_or_size_splits = x.shape[1]) )(inputs)

    v1 = refstc.model(keras.layers.concatenate([sc, pn, pb]))
    
    v2 = par.model(keras.layers.concatenate([pb, dp]))
    
    nmpiy = get_submodel(
        keras.layers.concatenate([v1, v2]), dim = 2
    )
    
    v3 = get_submodel(
        keras.layers.concatenate([sm, pb]), dim = 2
    )
    
    mpimodels = [
        rs.model('weights/mpimodel_allreduce.ckpt'),
        rs.model('weights/mpimodel_allgather.ckpt')
    ]
    
    mpimodel_results = []
    
    for m in mpimodels:
        # stop updating pretrained model
        m.trainable = False
    
    allreduce = mpimodels[0](
        keras.layers.concatenate([dp, pb]), training = False
    ) + smc - np.log(2) - np.log(10**6)
    
    allgather = mpimodels[1](
        keras.layers.concatenate([dp, pb]), training = False
    ) + smc - np.log(2) - np.log(10**6)
    
    y = keras.layers.Dense(1+1)(keras.layers.concatenate([nmpiy, v3, allreduce, allgather]))
    
    y = tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(
            loc=t[..., :1],
            scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))
    )(y)
    
    return keras.Model(inputs = inputs, outputs = y)

def logd(x):
    return np.log(x + 0.0001)

def predict(
    stc_code, stc_mpi, stc_mpic,
    problem_n, problem_nb,
    distr_procs, distr_hosts
):
    m = model()
    m.load_weights(MODEL_WEIGHTS)
    
    inputs = tf.constant([ [ logd(stc_code), logd(stc_mpi), logd(stc_mpic), logd(problem_n), logd(problem_nb), distr_procs, distr_hosts ]])
    
    pred = m(inputs)
    
    return (pred.mean().numpy()[0], pred.stddev().numpy()[0])

@tf.function
def model_loss_fn(y_true, y_pred):
    return -y_pred.log_prob(tf.cast(y_true, tf.float32)) + tf.losses.mean_squared_error(y_true, y_pred.mean())

@tf.function
def train_loop_simple(
    model, inputs, outputs, iters = 100, lr = 0.1
):
    loss_fn = model_loss_fn
    opt = tf.optimizers.AdaDelta(lr)
    for _ in range(iters):
        with tf.GradientTape() as tape:
            p = model(inputs)
            l = loss_fn(outputs, p)
        grads = tape.gradient(loss, model.train_variables)
        opt.apply_gradient(zip(grads, model.train_variables))
        

def increment(
    stc_code, stc_mpi, stc_mpic,
    problem_n, problem_nb,
    distr_procs, distr_hosts,
    result_time
):
    m = model()
    m.load_weights(MODEL_WEIGHTS)
    
    inputs = tf.constant([ [ logd(stc_code), logd(stc_mpi), logd(stc_mpic), logd(problem_n), logd(problem_nb), distr_procs, distr_hosts ]])
    outputs = tf.constant([ [result_time] ])
    
    train_loop_simple(m, inputs, outputs)
    
    m.save_weights(MODEL_WEIGHTS)
