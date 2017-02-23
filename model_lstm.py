# Model LSTM

import tflearn

def get_sgd():
	return tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)


def simple_lstm():
	input_layer = tflearn.input_data(shape=[None, 5, 19])
	lstm_layer = tflearn.lstm(input_layer, 128, dropout=0.8)
	fc = tflearn.fully_connected(lstm_layer, 11, activation='sigmoid')
	sgd = get_sgd()
	reg = tflearn.regression(fc, optimizer=sgd, learning_rate=0.001, loss='categorical_crossentropy')

	model = tflearn.DNN(reg, tensorboard_verbose=3)
	return model


def time_distributed_lstm():
	input_layer = tflearn.input_data(shape=[None, 13, 7])
	model = tflearn.time_distributed(input_layer, tflearn.lstm(128))
	model = tflearn.fully_connected(model, 7, activation='sigmoid')
	
	sgd = get_sgd()
	reg = tflearn.regression(model, optimizer=sgd, learning_rate=0.001, loss='categorical_crossentropy')
	
	model = tflearn.DNN(model, tensorboard_verbose=3)
	return model