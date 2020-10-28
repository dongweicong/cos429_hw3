# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.6.12 |Anaconda, Inc.| (default, Sep  9 2020, 00:29:25) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: layers/fn_conv.py
# Compiled at: 2020-10-07 20:45:04
# Size of source mod 2**32: 3250 bytes
import numpy as np, scipy.signal

def fn_conv(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [in_height] x [in_width] x [num_channels] x [batch_size] array
        params: Weight and bias information for the layer.
            params['W']: layer weights, [filter_height] x [filter_width] x [filter_depth] x [num_filters] array
            params['b']: layer bias, [num_filters] x 1 array
        hyper_params: Optional, could include information such as stride and padding.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [out_height] x [out_width] x [num_filters] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt weights, same size as params['W']
            grad['b']: gradient wrt bias, same size as params['b']
    """
    in_height, in_width, num_channels, batch_size = input.shape
    _, _, filter_depth, num_filters = params['W'].shape
    out_height = in_height - params['W'].shape[0] + 1
    out_width = in_width - params['W'].shape[1] + 1
    assert params['W'].shape[2] == input.shape[2], 'Filter depth does not match number of input channels'
    output = np.zeros([out_height, out_width, num_filters, batch_size])
    for i in range(batch_size):
        for j in range(num_filters):
            conv_im = np.zeros([out_height, out_width])
            for k in range(num_channels):
                filter = params['W'][:, :, k, j]
                im = input[:, :, k, i]
                conv_im = conv_im + scipy.signal.convolve(im, filter, 'valid')

            conv_im = conv_im + params['b'][j]
            output[:, :, j, i] = conv_im

    dv_input = np.zeros(0)
    grad = {'W':np.zeros(0),  'b':np.zeros(0)}
    if backprop:
        assert dv_output is not None
        dv_input = np.zeros(input.shape)
        grad['W'] = np.zeros(params['W'].shape)
        grad['b'] = np.zeros(params['b'].shape)
        for i in range(batch_size):
            for j in range(num_filters):
                for k in range(num_channels):
                    flipped_kernel = np.rot90(params['W'][:, :, k, j], 2)
                    dv_input[:, :, k, i] = dv_input[:, :, k, i] + scipy.signal.convolve(dv_output[:, :, j, i], flipped_kernel, 'full')
                    flipped_dv_out = np.rot90(dv_output[:, :, j, i], 2)
                    grad['W'][:, :, k, j] = grad['W'][:, :, k, j] + np.rot90(scipy.signal.convolve(input[:, :, k, i], flipped_dv_out, 'valid'), 2)

        grad['W'] = grad['W'] / batch_size
        grad['b'] = np.sum(dv_output, (0, 1, 3))[:, np.newaxis] / batch_size
    return (output, dv_input, grad)
# okay decompiling fn_conv_.pyc
