'''
	Tflearn classifier
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from datasetTools import load_dataset_from, prepare_as_2d, prepare_as_4d
from datasetTools import tf_simple_preparation, simple_preparation

from model_dnn import simple_dnn
from model_lstm import simple_lstm, time_distributed_lstm
from model_blstm import simple_blstm
from model_cnn import simple_cnn, three_branch_cnn, convnet, simple_cnn_2d
from model_elm import ELM

from random import randint

import time

import numpy


def train(model, dataset, name=None):
	[x_ref, x_test, y_ref, y_test] = data
	#average = 0
	#i = 0

	answer = 'n'

	print('x_test shape : ', x_ref.shape)
	print('x_test shape : ', x_test.shape)

	# Trainig model
	print('[+] Starting training !')
	start_time = time.time()
	model.fit(x_ref, y_ref, n_epoch=50, validation_set=None,
				show_metric=True, run_id=name)
	print("--- %s seconds ---" % (time.time() - start_time))
	
	# Evaluating model
	acc = model.evaluate(x_test, y_test, batch_size=2)
	print('[+] Model accuracy : ', acc[0]*100)

	name = name + ".tflearn"
	model.save(name)


	print('Do you want to demo the model on random sample ?')
	answer = input('[y/n] :')

	while answer == 'y':
		index = randint(1, 300)
		x_p = x_test[index]
		x_p = numpy.reshape(x_p, (1,5,19))
		y_p = y_test[index]
		
		model.predict(x_p)
		#print(result)


def demo(name):
	name = name + ".tflearn"
	model = model.load(name)



if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("model", 
			help="Choose a model type", 
			choices=["simple_dnn", "simple_lstm", "simple_blstm", "three_branch_cnn", 
					"simple_cnn","simple_cnn_2d","convnet", "time_distributed_lstm", "elm"])
	parser.add_argument("dataset_type", 
			help="Choose a dataset type : " +
				"0 : The dataset directory should contain 4 files as follow :"+
					" training data, training labels, test data, test labels"+
				"1 : The dataset directory contains all samples. data and labels are in the same.mat file.")
	args = parser.parse_args()

	# Fix random seed for reproducibility
	numpy.random.seed(7)

	# Raw path to dataset
	path = '/home/neurones/Documents/Developpement/Dataset/_3-2/'

	# Load dataset and prepare data
	if args.dataset_type == '0':
		data = load_dataset_from(path, int(args.dataset_type))
	if args.dataset_type == '1':
		path = '/home/neurones/Documents/Developpement/Dataset/_3-1/'
		data = load_dataset_from(path, int(args.dataset_type))
		
	
	'''
		DNN models
	'''
	if args.model == "simple_dnn":
		#data = tf_simple_preparation(data)
		model = simple_dnn()
		train(model, data, args.model)		
		exit()


	'''
		LSTM models
	'''
	if args.model == "simple_lstm":
		#data = simple_preparation(data)
		model = simple_lstm()
		name = args.model + '-ST'
		train(model, data, name=name)
		exit()

	elif args.model == "time_distributed_lstm":
		data = simple_preparation(data)
		model = time_distributed_lstm()
		train(model, data, name=args.model)
		exit()


	'''
		BLSTM models
	'''
	if args.model == "simple_blstm":
		#data = simple_preparation(data)
		model = simple_blstm()
		train(model, data, name=args.model)
		exit()


	'''
		CNN models
	'''
	if args.model == "simple_cnn":
		model = simple_cnn()
		#data = simple_preparation(data)
		name = args.model + '-ST'
		train(model, data, name=name)

	elif args.model == "simple_cnn_2d":
		data = prepare_as_4d(data)
		#data=simple_preparation(data)
		model = simple_cnn_2d()
		train(model, data, name=args.model)

	elif args.model == "three_branch_cnn":
		model = three_branch_cnn()
		train(model, data, name=args.model)

	elif args.model == "convnet":
		#data = prepare_as_4d(data)
		data=simple_preparation(data)
		model = convnet()
		train(model, data, name=args.model)


	'''
		ELM model
	'''
	if args.model == "elm":
		# Basic tf setting
		tf.set_random_seed(2016)
		sess = tf.Session()

		# Prepare data (as 2D matrix)
		#data = tf_simple_preparation(data)
		data = prepare_as_2d(data)
		x_ref, x_test, x_val, y_ref, y_test, y_val = data

		# Construct ELM
		num_train_ex, num_coeff = x_ref.shape

		batch_size = num_train_ex
		hidden_num = 100
		print("batch_size : {}".format(batch_size))
		print("hidden_num : {}".format(hidden_num))
		elm = ELM(sess, batch_size, num_coeff, hidden_num, 11)

		# one-step feed-forward training
		elm.feed(x_ref, y_ref)

		# testing
		elm.test(x_test, y_test)
