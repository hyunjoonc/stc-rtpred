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

mo = model.model()

mo.load_weights(model.MODEL_WEIGHTS)

# data setup. stc.mpic column is the state complexity of mpi collective (gather, scatter) routines.
input_columns = ['stc.code', 'stc.mpi', 'stc.mpic', 'problem.n', 'problem.nb', 'distr.procs', 'distr.hosts']
output_columns = ['result.time']

def data_preproc(df):
    for key in ['problem.n', 'problem.nb', 'stc.code', 'stc.mpi', 'stc.mpic']:
        if key in df:
            df[key] = np.log(df[key] + 0.001)
    
    return df

def err(true, pred, s = 1):
    pred_min = pred.mean() - s*pred.stddev()
    pred_max = pred.mean() + s*pred.stddev()
    
    return  1 - np.sum((pred_min <= true) & (true <= pred_max)) / len(true)

data = pd.read_csv('data/data.csv')
data = data_preproc(data)
input_data = data[input_columns]
output_data = data[output_columns]

# cutoff
sw1i = data['result.time'] > 4

plot_data = {
    'HPL': (input_data[(data['meta.config'] == 'hpl')].values, output_data[(data['meta.config'] == 'hpl')].values),
    'SW1': (input_data[(data['meta.config'] == 'sw1') & sw1i].values, output_data[(data['meta.config'] == 'sw1') & sw1i].values),
    'SW2': (input_data[(data['meta.config'] == 'sw2')].values, output_data[(data['meta.config'] == 'sw2')].values),
}

print (plot_data['SW1'][0], plot_data['SW1'][1])

plt.figure(figsize = (24,7))
plt.rc('font', size=24)
#plt.plot([1,50],[1,50])

i = 0
s = 1.2
for k, v in plot_data.items():
    i += 1
    
    test_input, test_output = v
    m, M = 0, max(test_output)
    z = mo(test_input)
    
    ax = plt.subplot(1, 3, i)
    ax2 = ax.twinx()
    
    error_bound_abcabc = np.maximum(test_output - (z.mean() + s * z.stddev()), (z.mean() - s * z.stddev()) - test_output)
    ax.plot([m,M], [m, M])
    ax.scatter(test_output, np.abs(z.mean()), color='tab:blue')
    
    error_bound_defdef = np.maximum(error_bound_abcabc, [0])
    iidx = error_bound_defdef > 0
    ax2.scatter(test_output[iidx], error_bound_defdef[iidx], color='tab:orange')
    
    #ax.scatter(test_output, z.mean() + s * z.stddev())
    #ax.scatter(test_output, z.mean() - s * z.stddev())
    ax.set_title('{} (E={:4.2f}%)'.format(k, 100*err(test_output, z, s)))#100 * error_rate(test_output, model(test_input).mean())))

    ax.set_xlabel('Actual (sec.)')
    ax.set_ylabel('Pred. mean (sec.)', color='tab:blue')
    ax2.set_ylabel('Range diff. (sec.)', color='tab:orange')
    
    #ax.legend(['mean', 'error'])#, '+{:.1f}s'.format(s), '-{:.1f}s'.format(s)]

#plt.subplots_adjust(wspace=0.3)
plt.tight_layout()
plt.savefig('r.png')