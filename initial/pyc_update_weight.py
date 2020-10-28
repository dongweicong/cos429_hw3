# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: /Users/jsu/Desktop/cos429_hw5_fall20/solution/update_weights.py
# Compiled at: 2020-10-07 20:51:32
# Size of source mod 2**32: 537 bytes
import numpy as np

def update_weights(model, grads, hyper_params):
    #print("pyc update weight")
    num_layers = len(grads)
    a = hyper_params['learning_rate']
    lmd = hyper_params['weight_decay']
    updated_model = model
    for i in range(num_layers):
        W = model['layers'][i]['params']['W']
        b = model['layers'][i]['params']['b']
        # print("1) ",a *grads[i]['W'] )
        # print("2) ",lmd * W )
        W = W - (a * grads[i]['W'] + lmd * W)
        b = b - a * grads[i]['b']
        updated_model['layers'][i]['params']['W'] = W
        updated_model['layers'][i]['params']['b'] = b

    return updated_model
# okay decompiling update_weights_.pyc
