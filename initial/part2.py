"""
Basic script to create a new network model.
The model presented here is meaningless, but it shows how to properly 
call init_model and init_layers for the various layer types.
"""

import sys
sys.path += ['layers']
import numpy as np
from init_layers import init_layers
from init_model import init_model
# from inference import inference
# from loss_euclidean import loss_euclidean

from load_MNIST_images import load_MNIST_images
from load_MNIST_labels import load_MNIST_labels
from train import train

def main():

    # Load training data
    train_data = load_MNIST_images('train-images.idx3-ubyte')
    # print (train_data.shape)
    train_label = load_MNIST_labels('train-labels.idx1-ubyte')
    # Load testing data
    test_data = load_MNIST_images('t10k-images.idx3-ubyte')
    test_label = load_MNIST_labels('t10k-labels.idx1-ubyte')

    trainlen = len(train_data)

    l = [init_layers('conv', {'filter_size': 2,
                              'filter_depth': 1,
                              'num_filters': 6}),
         init_layers('pool', {'filter_size': 2,
                              'stride': 2}),
         init_layers('conv', {'filter_size': 3,
                              'filter_depth': 6,
                              'num_filters': 16}),
         init_layers('relu', {}),
         init_layers('flatten', {}),
         init_layers('linear', {'num_in': 338,
                                'num_out': 10}),
         init_layers('softmax', {})]

    params = {
        "learning_rate": .01,
        "weight_decay": .0005,
        "batch_size": 128,
        "save_file": 'model.npz'
    }

    model = init_model(l, train_data.shape, 10, True)

    model, loss = train(model, train_data, train_label, params, numIters)

if __name__ == '__main__':
    main()
