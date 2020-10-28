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

    num_nodes, batch_size = input.shape
    exp_input = np.exp(input)

    # Initialize
    output = np.zeros([num_nodes, batch_size])
    dv_input = np.zeros(0)
    # grad is included to maintain consistency in the return values of layers,
    # but there is no gradient to calculate in the softmax layer since there
    # are no weights to update.
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}

    # TODO: FORWARD CODE
    #       Update output with values
    sums = np.array([exp_input.sum(axis = 0)])
    output = exp_input/sums


    if backprop:
        assert dv_output is not None
        dv_input = np.zeros([num_nodes, batch_size])

        # TODO: BACKPROP CODE
        #       Update dv_input with values
        jacobian = np.zeros([num_nodes, num_nodes, batch_size])
        for i in range (batch_size):
            output_ith = np.reshape(output[:, i], (num_nodes,1))
            jacobian[:,:,i] = np.matmul(-output_ith, output_ith.T)
            # The above general formula only works for non-diagonal entry. For diagonal entry:
            for j in range (num_nodes):
                jacobian[(j,j,i)] = output[(j, i)] * (1 - output[(j, i)])
            dv_input[:, i] = np.matmul(jacobian[:, :, i], dv_output[:, i])


    return output, dv_input, grad
