import numpy as np

def inference(model, input):
    """
    Do forward propagation through the network to get the activation
    at each layer, and the final output
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
    Returns:
        output: The final output of the model
        activations: A list of activations for each layer in model["layers"]
    """
    # print("Now in the inference function")
    num_layers = len(model['layers'])
    activations = [None,] * num_layers

    # TODO: FORWARD PROPAGATION CODE
    input_layer = input

    for i in range (num_layers):
        layer = model['layers'][i]
        if i != 0:
            input_layer = activations[i-1]
        activation, dv_input, grad = layer['fwd_fn'](input_layer, layer['params'], layer['hyper_params'], False, None)
        activations[i] = activation

    output = activations[-1]
    return output, activations
