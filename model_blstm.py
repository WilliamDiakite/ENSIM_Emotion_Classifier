# BLSTM Model

import tflearn as tf

def simple_blstm():
	input_layer = tf.input_data(shape=[None, 5, 19])
	model = tf.bidirectional_rnn(input_layer, tf.BasicLSTMCell(91), tf.BasicLSTMCell(91))
	model = tf.dropout(model, 0.5)
	model = tf.fully_connected(model, 11, activation='sigmoid')

	sgd = tf.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=1000)
	model = tf.regression(model, optimizer=sgd, loss='categorical_crossentropy')

	return tf.DNN(model, clip_gradients=0., tensorboard_verbose=0)

