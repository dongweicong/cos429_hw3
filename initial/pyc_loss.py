# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.6.12 |Anaconda, Inc.| (default, Sep  9 2020, 00:29:25) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: layers/loss_crossentropy.py
# Compiled at: 2020-10-04 15:08:38
# Size of source mod 2**32: 976 bytes
import numpy as np

def loss_crossentropy(input, labels, hyper_params, backprop):
    """
    Args:
        input: [num_nodes] x [batch_size] array
        labels: [batch_size] array
        hyper_params: Dummy input. This is included to maintain consistency across all layer and loss functions, but the input argument is not used.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.

    Returns:
        loss: scalar value
        dv_input: The derivative of the loss with respect to the input. Same size as input.
    """
    #print("pyc loss")
    assert labels.max() < input.shape[0]
    batch_size = labels.size
    ind0 = np.squeeze(labels.astype('i'))
    ind1 = np.arange(input.shape[1])
    loss = -np.sum(np.log(input[(ind0, ind1)])) / batch_size
    dv_input = np.zeros(0)
    eps = 1e-05
    if backprop:
        dv_input = np.zeros(input.shape)
        dv_input[(ind0, ind1)] = -1 / (input[(ind0, ind1)] + eps)
    return (loss, dv_input)
# okay decompiling loss_crossentropy_.pyc
