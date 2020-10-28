import numpy as np

def calc_gradient(model, input, layer_acts, dv_output):
    '''
    Calculate the gradient at each layer, to do this you need dv_output
    determined by your loss function and the activations of each layer.
    The loop of this function will look very similar to the code from
    inference, just looping in reverse.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
        layer_acts: A list of activations of each layer in model["layers"]
        dv_output: The partial derivative of the loss with respect to each element in the output matrix of the last layer.
    Returns:
        grads:  A list of gradients of each layer in model["layers"]
    '''
    num_layers = len(model["layers"])
    grads = [None,] * num_layers

    # TODO: Determine the gradient at each layer.
    #       Remember that back-propagation traverses
    #       the model in the reverse order.

    for i in range (num_layers-1, 0,-1):
        layer = model['layers'][i]
        input_layer = layer_acts[i-1]
        # print("The dimension of input layer is: ", input_layer.shape)
        _, dv_output, grad = layer['fwd_fn'](input_layer, layer['params'], layer['hyper_params'], True, dv_output)
        grads[i] = grad

    layer = model['layers'][0]
    input_layer = input
    _, dv_output, grad = layer['fwd_fn'](input_layer, layer['params'], layer['hyper_params'], True, dv_output)
    grads[0] = grad
    return grads
