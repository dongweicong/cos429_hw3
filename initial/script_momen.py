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
from init_model import init_model

# # Load training data
# train_data = load_MNIST_images('train-images.idx3-ubyte')
# print (train_data.shape)
train_label = load_MNIST_labels('train-labels.idx1-ubyte')
# # Load testing data
# test_data = load_MNIST_images('t10k-images.idx3-ubyte')
test_label = load_MNIST_labels('t10k-labels.idx1-ubyte')

train_data=np.load("resized_train_data.npy")
test_data=np.load("resized_test_data.npy")





##############################resize#####################

# lens = train_data.shape[3]
# new_train_data = np.zeros((32,32,1,60000))
# for i in range (lens):
# 	new_train_data[:,:,0,i] = cv2.resize(train_data[:,:,0,i], (32, 32))
#
# # resize testing_data
# lent = test_data.shape[3]
# new_test_data = np.zeros((32,32,1,test_data.shape[3]))
# for i in range (lent):
# 	new_test_data[:,:,0,i] = cv2.resize(test_data[:,:,0,i], (32, 32))
#
# np.save("resized_train_data",new_train_data)
# np.save("resized_test_data",new_test_data)

#############################################################


l = [init_layers('conv', {'filter_size': 5,
						  'filter_depth': 1,
						  'num_filters': 6}),
	 init_layers('pool', {'filter_size': 2,
						  'stride': 2}),
	 init_layers('conv', {'filter_size': 5,
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
	"learning_rate": .02,
	"weight_decay": .0005,
	"batch_size": 300,
	"save_file": 'model.npz'
}

model = init_model(l, list(train_data[:,:,:,0].shape), 10, True)


input = train_data
numIters = 2000
rho = 0.9

# model = np.load('potential_highacc_model_danlu323.npz', allow_pickle=True)
# model = dict(model)

label = train_label

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
update_params = {"learning_rate": lr,
				  "weight_decay": wd }

num_inputs = input.shape[-1]

# This is saved for plotting
loss = np.zeros((numIters,))
loss_test = np.zeros(((numIters//25)+1,))

# This is the BATCH accuracy (WARNING: DO NOT DELETE)
batch_accuracy = np.zeros((numIters,))

# This is saved for testing
# (WARNING2: DO NOT DELETE, saved for plotting)
testaccuracy = np.zeros(((numIters//25)+1,))

# this is for save model
maxmax = 0.0


for i in range(numIters):
#   lr = lr_initial * np.exp(-np.log(2)/1500*i)
	batch_num = np.random.randint(num_inputs, size=batch_size)
	sample_input = input[:,:,:,batch_num]

	sample_label = label[batch_num]
	sample_output, sample_activations = inference(model, sample_input)
	loss[i], dv_gradient = loss_crossentropy(sample_output, sample_label, {}, True)
	count = 0.0
	for j in range (batch_size):
		if (np.argmax(sample_output[:,j]) == sample_label[j]):
			count +=1.0
	batch_accuracy[i] = count/batch_size
	#print("Epoch: ", i, "loss: ", loss[i],"The batch accuracy is: ", batch_accuracy[i])

	# NEW: TESTING AND SAVE EVERY 25 ITERATIONS
	test_size=1000
	if (i % 25 == 0):
		test_num = np.random.randint(test_data.shape[-1], size=test_size)
		sample_test_data = test_data[:,:,:,test_num]
		sample_test_label = test_label[test_num]
		test_output, activations = inference(model, sample_test_data)
		loss_test[(i//25)], _ = loss_crossentropy(test_output, sample_test_label, {}, False)
		count=0.0
		for k in range (test_size):
			if (np.argmax(test_output[:,k]) == sample_test_label[k]):
				count +=1.0
		testaccuracy[(i//25)] = count/test_size
		print("Test loss is : ",loss_test[(i//25)],"The test accuracy is: ", count/test_size)
		# NOW TIME TO SAVE MODEL: if it's better then save it
		if (maxmax <= testaccuracy[(i//25)]):
			maxmax = testaccuracy[(i//25)]
			np.savez("momen/max_model3.npz", **model)
			np.save("momen/min_loss",loss)

	sample_grads = calc_gradient(model, sample_input, sample_activations, dv_gradient)

	# implements momentum (basic momentum)
	num_layers = len(model['layers'])
	for s in range(num_layers):
		if (i == 0):
			sample_grads[s]['W'] -= 0.0 * sample_grads[s]['W']
		else:
			sample_grads[s]['W'] -= lr* rho * sample_grads[s]['W']

	model = update_weights (model, sample_grads, update_params)

np.save("momen/test_loss_curve", loss)
np.save("momen/train_loss_curve", loss)
np.save("momen/batches_accuracys", batch_accuracy)
np.save("momen/test_accuracy", testaccuracy)
