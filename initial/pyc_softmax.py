# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.6.12 |Anaconda, Inc.| (default, Sep  9 2020, 00:29:25) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: layers/fn_softmax.py
# Compiled at: 2020-10-07 20:45:14
# Size of source mod 2**32: 1783 bytes
import numpy as np

def fn_softmax(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [num_nodes] x [batch_size] array
        params: Dummy input. This is included to maintain consistency across all layers, but the input argument is not used.
        hyper_params: Dummy input. This is included to maintain consistency across all layers, but the input argument is not used.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [num_nodes] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: Dummy output. This is included to maintain consistency in the return values of layers, but there is no gradient to calculate in the softmax layer since there are no weights to update.
    """
    #print("pyc softmax")
    num_nodes, batch_size = input.shape
    exp_input = np.exp(input)
    output = exp_input / np.sum(exp_input, 0, keepdims=True)
    dv_input = np.zeros(0)
    grad = {'W':np.zeros(0),  'b':np.zeros(0)}
    if backprop:
        assert dv_output is not None
        m = np.zeros([num_nodes, num_nodes, batch_size])
        dv_input = np.zeros([num_nodes, batch_size])
        for i in range(batch_size):
            m[:, :, i] = -output[:, i:i + 1] @ output[:, i:i + 1].T
            for j in range(num_nodes):
                m[(j, j, i)] = output[(j, i)] * (1 - output[(j, i)])

            dv_input[:, i] = m[:, :, i] @ dv_output[:, i]

    return (
     output, dv_input, grad)
# okay decompiling fn_softmax_.pyc
