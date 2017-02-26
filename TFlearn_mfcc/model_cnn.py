# CNN Model

import tflearn as tf


def simple_cnn():
	input_layer = tf.input_data(shape=[None, 13, 7])
	model = tf.conv_1d(input_layer, 256, 4, padding='valid', activation='sigmoid', regularizer='L2')
	model = tf.max_pool_1d(model, kernel_size=4)
	model = tf.dropout(model, 0.7)
	model = tf.fully_connected(model, 7, activation='sigmoid')

	sgd = tf.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=32000)
	model = tf.regression(model, optimizer=sgd, loss='categorical_crossentropy')

	return tf.DNN(model)


def simple_cnn_2d():
	input_layer = tf.input_data(shape=[None, 5, 19, 1])
	model = tf.conv_2d(input_layer, 256, 3, activation='sigmoid', regularizer='L2')
	model = tf.max_pool_2d(model, 2)
	'''
	model = tf.local_response_normalization(model)
	model = tf.conv_2d(model, 512, 3, padding='valid', activation='sigmoid', regularizer='L2')
	model = tf.max_pool_2d(model, 2)
	model = tf.local_response_normalization(model)
	'''
	model = tf.dropout(model, 0.7)
	model = tf.fully_connected(model, 11, activation='sigmoid')
	#model = tf.dropout(model, 0.7)
	
	#sgd = tf.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
	mom = tf.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
	model = tf.regression(model, optimizer=mom, loss='categorical_crossentropy')

	return tf.DNN(model, tensorboard_verbose=0)



def convnet():
	model = tf.input_data(shape=[None, 5, 19, 5], name='input')
	
	model = tf.conv_2d(model, 32, 3, activation='relu', regularizer="L2")
	model = tf.max_pool_2d(model, 2)
	model = tf.local_response_normalization(model)
	model = tf.conv_2d(model, 64, 3, activation='relu', regularizer="L2")
	model = tf.max_pool_2d(model, 2)
	model = tf.local_response_normalization(model)
	model = tf.fully_connected(model, 128, activation='tanh')
	model = tf.dropout(model, 0.8)
	model = tf.fully_connected(model, 256, activation='tanh')
	model = tf.dropout(model, 0.8)
	model = tf.fully_connected(model, 11, activation='sigmoid')
	model = tf.regression(model, optimizer='adam', learning_rate=0.01,
							loss='categorical_crossentropy', name='target')

	model = tf.DNN(model, tensorboard_verbose=0)
	return model


def three_branch_cnn():
	input_layer = tf.input_data(shape=[None, 5, 19])

	# Branching...
	branch1 = tf.conv_1d(input_layer, 256, 5, padding='valid', activation='sigmoid', regularizer='L2')
	
	branch2 = tf.conv_1d(input_layer, 256, 5, padding='valid', activation='sigmoid', regularizer='L2')
	
	branch3 = tf.conv_1d(input_layer, 256, 5, padding='valid', activation='sigmoid', regularizer='L2')
	
	# Merging
	model = tf.merge([branch1, branch2, branch3], mode='concat', axis=1)
	#model = tf.max_pool_1d(model, kernel_size=5)
	model = tf.dropout(model, 0.7)
	model = tf.fully_connected(model, 11, activation='sigmoid')

	sgd = tf.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=1000)
	model = tf.regression(model, optimizer='sgd', loss='categorical_crossentropy')

	return tf.DNN(model, tensorboard_verbose=0)

