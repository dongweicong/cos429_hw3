import numpy as np
from pyc_linear import fn_linear
from fn_flatten import fn_flatten
from fn_relu import fn_relu
from fn_pool import fn_pool
from pyc_conv import fn_conv
from pyc_softmax import fn_softmax
from pyc_loss import loss_crossentropy

from pyc_cal_grad import calc_gradient
from pyc_inference import inference
from init_layers import init_layers
from train import train
from pyc_update_weight import update_weights
from init_model import init_model

import sys
sys.path += ['layers']
# from loss_euclidean import loss_euclidean

from load_MNIST_images import load_MNIST_images
from load_MNIST_labels import load_MNIST_labels

# Load training data
train_data = load_MNIST_images('train-images.idx3-ubyte')
print (train_data.shape)
train_label = load_MNIST_labels('train-labels.idx1-ubyte')
# Load testing data
test_data = load_MNIST_images('t10k-images.idx3-ubyte')
test_label = load_MNIST_labels('t10k-labels.idx1-ubyte')

trainshape = train_data.shape
num_in = trainshape[0]*trainshape[1]*trainshape[2]*trainshape[3]

l = [init_layers('conv', {'filter_size': 2,
                          'filter_depth': 1,
                          'num_filters': 6}),
     init_layers('pool', {'filter_size': 2,
                          'stride': 2}),
     init_layers('conv', {'filter_size': 4,
                          'filter_depth': 6,
                          'num_filters': 16}),
     init_layers('pool', {'filter_size': 2,
                          'stride': 2}),
     init_layers('relu', {}),
     init_layers('flatten', {}),
     init_layers('linear', {'num_in': 400,
                    'num_out': 120}),
    init_layers('linear', {'num_in': 120,
                    'num_out': 84}),
     init_layers('linear', {'num_in': 84,
                            'num_out': 10}),
     init_layers('softmax', {})]

params = {
#     "learning_rate": .01,
#     "weight_decay": .0005,
    "learning_rate": .02,
    "weight_decay": .0005,
    "batch_size": 300,
    "save_file": 'model.npz'
}

model = init_model(l, list(train_data[:,:,:,0].shape), 10, True)

input = train_data
numIters = 5000
label = train_label
max_model=None
max_acc=0


lr = params.get("learning_rate", .01)
# Weight decay
wd = params.get("weight_decay", .0005)
# Batch size
batch_size = params.get("batch_size", 300)
# There is a good chance you will want to save your network model during/after
# training. It is up to you where you save and how often you choose to back up
# your model. By default the code saves the model in 'model.npz'.
save_file = params.get("save_file", 'model.npz')

# update_params will be passed to your update_weights function.
# This allows flexibility in case you want to implement extra features like momentum.
update_params = {"learning_rate": lr,
                 "weight_decay": wd }

print("Learning Rate: ", lr)

num_inputs = input.shape[-1]
loss = np.zeros((numIters,))

for i in range(numIters):
    # TODO: One training iteration
    # Steps:
    #   (1) Select a subset of the input to use as a batch
    #   (2) Run inference on the batch
    #   (3) Calculate loss and determine accuracy
    #   (4) Calculate gradients
    #   (5) Update the weights of the model
    # Optionally,
    #   (1) Monitor the progress of training
    #   (2) Save your learnt model, using ``np.savez(save_file, **model)``
    batch_num = np.random.randint(num_inputs, size=batch_size)
    sample_input = input[:,:,:,batch_num]

    sample_label = label[batch_num]
    sample_output, sample_activations = inference(model, sample_input)
    loss[i], dv_gradient = loss_crossentropy(sample_output, sample_label, {}, True)



    count = 0.0
    for j in range (batch_size):
        if (np.argmax(sample_output[:,j]) == sample_label[j]):
            count +=1.0
    print("Epoch: ", i, " loss: ", loss[i], " The accuracy is: ", count/batch_size)
    if count/batch_size>0.97:
        np.savez("script1/model"+str(i)+".npz", **model)



    sample_grads = calc_gradient(model, sample_input, sample_activations, dv_gradient)
    model = update_weights (model, sample_grads, update_params)
