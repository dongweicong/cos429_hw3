# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.6.12 |Anaconda, Inc.| (default, Sep  9 2020, 00:29:25) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: layers/fn_linear.py
# Compiled at: 2020-10-03 21:48:19
# Size of source mod 2**32: 1928 bytes
import numpy as np

def fn_linear(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [num_in] x [batch_size] array
        params: Weight and bias information for the layer.
            params['W']: layer weights, [num_out] x [num_in] array
            params['b']: layer bias, [num_out] x 1 array
        hyper_params: Information describing the layer.
            hyper_params['num_in']: number of inputs for layer
            hyper_params['num_out']: number of outputs for layer
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [num_out] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt weights, same size as params['W']
            grad['b']: gradient wrt bias, same size as params['b']
    """
    #print("pyc linear")
    W = params['W']
    b = params['b']

    num_in, batch_size = input.shape
    if num_in != hyper_params['num_in']:
        print('Incorrect number of inputs provided at linear layer.\n Got %d inputs,  expected %d.' % num_in, hyper_params['num_in'])
        raise

    output = W @ input + b
    dv_input = np.zeros(0)
    grad = {'W':np.zeros(0),  'b':np.zeros(0)}

    if backprop:
        assert dv_input is not None
        dv_input = W.T @ dv_output
        grad['W'] = dv_output @ input.T / batch_size
        grad['b'] = np.sum(dv_output, 1, keepdims=True) / batch_size
    return (output, dv_input, grad)
# okay decompiling fn_linear_.pyc
