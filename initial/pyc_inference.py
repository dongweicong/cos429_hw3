# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: /Users/jsu/Desktop/cos429_hw5_fall20/solution/inference.py
# Compiled at: 2020-10-07 20:47:55
# Size of source mod 2**32: 625 bytes
import numpy as np

def inference(model, input):
    """
    Given an input, perform inference and produce an output
    assuming the network is a chain.
    """
    #print("pyc inference")
    num_layers = len(model['layers'])
    activations = [None] * num_layers
    for i in range(num_layers):
        layer = model['layers'][i]
        if i == 0:
            layer_in = input
        else:
            layer_in = activations[(i - 1)]
        activation, _, _ = layer['fwd_fn'](layer_in, layer['params'], layer['hyper_params'], False, None)
        activations[i] = activation

    output = activations[(-1)]
    return (output, activations)
# okay decompiling inference_.pyc
