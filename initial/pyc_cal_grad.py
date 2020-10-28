# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: /Users/jsu/Desktop/cos429_hw5_fall20/solution/calc_gradient.py
# Compiled at: 2020-10-03 22:47:15
# Size of source mod 2**32: 675 bytes
import numpy as np

def calc_gradient(model, input, layer_acts, dv_output):
    #print("pyc cal grad")
    num_layers = len(model['layers'])
    grads = [None] * num_layers
    for i in reversed(range(num_layers)):
        layer = model['layers'][i]
        if i == 0:
            layer_in = input
        else:
            layer_in = layer_acts[(i - 1)]
        _, dv_output, grad = layer['fwd_fn'](layer_in, layer['params'], layer['hyper_params'], True, dv_output)
        grads[i] = grad

    return grads
# okay decompiling calc_gradient_.pyc
