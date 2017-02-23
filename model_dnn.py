# Simple DNN with TFlearn 

from __future__ import division, print_function, absolute_import

import tflearn


def simple_dnn():
	input_layer = tflearn.input_data(shape=[None,5,19])

	# 1st hidden layer
	dense1 = tflearn.fully_connected(input_layer, 128, activation='tanh', 
									regularizer='L2', weight_decay=0.001)
	dropout1 = tflearn.dropout(dense1, 0.5)

	# 2nd hidden layer
	dense2 = tflearn.fully_connected(dropout1, 256, activation='tanh', 
									regularizer='L2', weight_decay=0.001)
	dropout2 = tflearn.dropout(dense2, 0.5)

	# Activation layer
	softmax = tflearn.fully_connected(dropout2, 11, activation='sigmoid')

	# Regression
	sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
	top_k = tflearn.metrics.Top_k(1)

	model = tflearn.regression(softmax, optimizer=sgd, loss='categorical_crossentropy')

	dnn_model = tflearn.DNN(model, tensorboard_verbose=3)
	return dnn_model




