import numpy as np

def update_weights(model, grads, hyper_params):
    # constant rho = 0.9, might need to adjust later
    rho = 0.9
    vx = 0.0
    old_v = vx
    '''
    Update the weights of each layer in your model based on the calculated gradients
    Args:
        model: Dictionary holding the model
        grads: A list of gradients of each layer in model["layers"]
        hyper_params:
            hyper_params['learning_rate']
            hyper_params['weight_decay']: Should be applied to W only.
    Returns:
        updated_model:  Dictionary holding the updated model
    '''
    num_layers = len(grads)
    a = hyper_params["learning_rate"]
    lmd = hyper_params["weight_decay"]
    updated_model = model

    # TODO: Update the weights of each layer in your model based on the calculated gradients
    for i in range(num_layers):
        b = model['layers'][i]['params']['b']
        W = model['layers'][i]['params']['W']

        old_v = vx
        # print("1) ",rho*vx)
        #
        # print("2) ",a *grads[i]['W'] )
        # print("dim ",(a *grads[i]['W']).shape)
        # print("3) ",rho * vx - a *grads[i]['W'])
        vx = rho * vx - a *grads[i]['W']
        W += -rho * old_v + (1+ rho) * vx - lmd * W
        b -= a * grads[i]['b']
        updated_model['layers'][i]['params']['W'] = W
        updated_model['layers'][i]['params']['b'] = b

    return updated_model
