import numpy 
from numpy import argmax
from numpy import transpose, reshape, zeros

from random import shuffle

import scipy
from scipy import io
from PIL import Image

from keras.utils import np_utils

import os
from sys import exit



'''
def save_best_config(model_accuracy):
	if model_accuracy > 25:
		file = open('best_results.txt', 'r+')
		file.seek(0,2)
		file.write(('Model Accuracy : '+ str(model_accuracy)))
		file.write(get_config())
'''


'''
	This function loads the dataset available in current directory
'''
def load_dataset_from(path_directory, folder_type):
	print('[...] Loading data and creating datasets')

	# if dataset contains only 4 files
	if os.path.isdir(path_directory):

		'''
			# If there is four seperate txt files
		'''
		if folder_type == 0:
			if len(os.listdir(path_directory)) == 4:
				print('\n[+] Loading files : \n {} \n'.format(os.listdir(path_directory)))
				x_ref 	= numpy.loadtxt(path_directory + 'CoefficientsObjet2Ref.txt')
				x_test 	= numpy.loadtxt(path_directory + 'CoefficientsObjet2Test.txt')
				y_ref 	= numpy.loadtxt(path_directory + 'ClasseObjet2Ref.txt')
				y_test 	= numpy.loadtxt(path_directory + 'ClasseObjet2Test.txt')

				print('x_test shape ', x_test.shape)
				print('x_ref shape ', x_ref.shape)
				return [x_ref, x_test, y_ref, y_test]
			else:
				print('[!] Error : {nbfiles} found in directory. Should be 4.'
					.format(nbfiles=len(os.listdir(path_directory))))	
				exit()

		'''	
			if every samples are in directory
		'''
		if folder_type == 1:

			# Fix random seed for reproducibility
			#numpy.random.seed(42)

			# Shuffle list of files in directory to init training, test and validations sets
			fileList = os.listdir(path_directory)[:]
			shuffle(fileList)

			# Training set ~ 60%, Test & Validation set ~ 20%
			n_total_sample = len(fileList)
			n_train_sample = int((n_total_sample*80)/100)
			#temp = n_total_sample - n_train_sample
			#n_test_sample = int(temp/2)
			#n_val_sample = temp - n_test_sample
			n_test_sample = n_total_sample - n_train_sample
			
			'''
			print('n_total_sample : ', n_total_sample)
			print('n_train_sample : ', n_train_sample)
			print('n_test_sample : ', n_test_sample)
			print('n_val_sample : ', n_val_sample)
			'''

			# Dataset initialization
			x_train = zeros((n_train_sample, 5, 19))
			x_test = zeros((n_test_sample, 5, 19))
			#x_val = zeros((n_val_sample, 5, 19))
			y_train = y_test = y_val = zeros((11, 1))

			# Create training set
			for i in range(0, n_train_sample):
				full_name = path_directory + '/' + fileList[i]
				current_sample = scipy.io.loadmat(full_name)
				x_train[i] = reshape(numpy.asarray(current_sample['Vec']), (5,19))
				y_train = numpy.concatenate((y_train, numpy.asarray(current_sample['Class'])), axis=1)
			# Removing first row because of zero initialization
			y_train = numpy.delete(y_train, 0, 1)
			y_train = y_train.T
				
			# Create test set
			for i in range(n_train_sample, n_test_sample + n_train_sample):
				index = i-n_train_sample
				full_name = path_directory + '/' + fileList[i]
				current_sample = scipy.io.loadmat(full_name)
				x_test[index] = reshape(numpy.asarray(current_sample['Vec']), (5,19))
				y_test = numpy.concatenate((y_test, numpy.asarray(current_sample['Class'])), axis=1)
			# Removing first row because of zero initialization
			y_test = numpy.delete(y_test, 0, 1)
			y_test = y_test.T

			'''
			# Create validation set
			for i in range(n_test_sample + n_train_sample, n_val_sample + n_test_sample + n_train_sample-1):
				index = i- (n_test_sample+n_train_sample)
				full_name = path_directory + '/' + fileList[i]
				current_sample = scipy.io.loadmat(full_name)
				x_val[index] = reshape(numpy.asarray(current_sample['Vec']), (5,19))
				y_val = numpy.concatenate((y_val, numpy.asarray(current_sample['Class'])), axis=1)
			# Removing first row because of zero initialization
			y_val = numpy.delete(y_val, 0, 1)
			y_val = y_val.T
			
			
			# Going through all files and creatig two big np arrays
			for filename in fileList:
				if filename.endswith(".mat"):
					full_name = directory + '/' + filename
					current_sample = scipy.io.loadmat(full_name)
					x[i] = reshape(numpy.asarray(current_sample['Vec']), (5,19))
					y = numpy.concatenate((y, numpy.asarray(current_sample['Class'])), axis=1)
					i += 1
				else:
					continue

			print('[+] ---- Shape of Data ----')
			print('[+] Training set 	: ', x_train.shape)
			print('[+] Taining labels 	: ', y_train.shape)
			print('[+] Test set 		: ', x_test.shape)
			print('[+] Test labels 	: ', y_test.shape)
			print('[+] Validation set 	: ', x_val.shape)
			print('[+] Validation labels   : ', y_val.shape)
			'''

			#return [x_train, x_test, x_val, y_train, y_test, y_val]
			return [x_train, x_test, y_train, y_test]

	else:
			print('[!] Argument is not a path to directory.')
			exit()



'''
	Create training set, test set & validation set

def init_dataset(data, labels):
	n_sample, lig, col = data.shape
	dummy, emotions = labels.shape

	# Nb of training samples should be about 60% of the total
	n_train = int((n_sample*60)/100)

	# Create training set 
	x_train = zeros((n_train, lig, col))
	y_train = zeros((n_train, emotions))
	for i in range(0, n_train):
		x_train[i] = data[i]
		y_train[i] = labels[i]


	# Nb of test samples should be about 20% of total
	n_test = int((n_sample*20)/100)

	# Create test set
	x_test = zeros((n_test, lig, col))
	y_test = zeros((n_test, emotions))
	for i in range(n_train)
'''

#print('Class : ', current_sample['Class'].shape)
#print('Data : ', current_sample['Vec'].shape)
#print(current_sample['Class'])



'''
	First preparation of the dataset transposing matrixes
'''
def simple_preparation(dataset):

	[x_ref, x_test, x_val, y_ref, y_test, y_val] = dataset
	#[x_ref, x_test, y_ref, y_test] = dataset
	
	# Get number of coefficients
	n_train_sample, ligt, colt = x_ref.shape
	n_test_sample, lig, col = x_test.shape
	n_val_sample, lig, col = x_val.shape
	num_coef = lig*col

	#n_train_sample, lig,  = x_ref.shape
	#n_test_sample, lig = x_test.shape



	print('[+] Transposing and reshaping all matrixes...')
	x_ref 	= x_ref.transpose()
	x_ref	= reshape(x_ref, (n_train_sample,5,19))
	x_test 	= x_test.transpose()
	x_test	= reshape(x_test, (n_test_sample,5,19))
	x_val	= x_val.transpose()
	x_val	= reshape(x_val, (n_val_sample,5,19))
	'''
	y_ref 	= y_ref.transpose()
	y_test	= y_test.transpose()
	y_val	= y_val.transpose()
	'''
	y_ref 	= y_ref.argmax(1)
	y_test	= y_test.argmax(1)
	y_val	= y_val.argmax(1)

	y_ref = np_utils.to_categorical(y_ref, 11)
	y_test = np_utils.to_categorical(y_test, 11)
	y_val = np_utils.to_categorical(y_val, 11)	
	
	print('\tDone ! \n')

	print('[+] New size of matrixes : \n x_ref : {xr} \n x_test : {xt} \n y_ref : {yr} \n y_test : {yt} \n'
		.format(xr=x_ref.shape, xt=x_test.shape, yr=y_ref.shape, yt=y_test.shape))

	#print('1st x_ref sample (should be shape (13,7)) : \n', x_ref[0])
	#print('1st y_ref value : \n', y_ref[0])

	return [x_ref, x_test, x_val, y_ref, y_test, y_val]
	#return [x_ref, x_test, y_ref, y_test]


'''
	Prepare data as 2D matrices
'''
def prepare_as_2d(dataset):
	x_ref, x_test, x_val, y_ref, y_test, y_val = dataset

	# Get number of coefficients
	n_train_sample, lig, col = x_ref.shape
	n_test_sample, lig, col = x_test.shape
	n_val_sample, lig, col = x_val.shape
	num_coef = lig*col

	# Reshape data as 2d matrices
	x_ref = reshape(x_ref, (n_train_sample, num_coef))
	x_test = reshape(x_test, (n_test_sample, num_coef))
	x_val = reshape(x_val, (n_val_sample, num_coef))
	
	print('[+] ----- Data Shape -----')
	print('[+] Training set 	: ', x_ref.shape)
	print('[+] Taining labels 	: ', y_ref	.shape)
	print('[+] Test set 		: ', x_test.shape)
	print('[+] Test labels 	: ', y_test.shape)
	print('[+] Validation set 	: ', x_val.shape)
	print('[+] Validation labels   : ', y_val.shape)

	return x_ref, x_test, x_val, y_ref, y_test, y_val






'''
	Prepare as 4d matrices (usefull for 2d conv)
'''
def prepare_as_4d(dataset):
	x_ref, x_test, x_val, y_ref, y_test, y_val = dataset

	# Get number of coefficients
	n_train_sample, ligt, colt = x_ref.shape
	n_test_sample, lig, col = x_test.shape
	n_val_sample, lig, col = x_val.shape
	num_coef = lig*col

	# Reshape data as 2d matrices
	x_ref = reshape(x_ref, (n_train_sample, ligt, colt, 1))
	x_test = reshape(x_test, (n_test_sample, lig, col, 1))
	x_val = reshape(x_val, (n_val_sample, lig, col, 1))
	

	print('[+] ----- Data Shape -----')
	print('[+] Training set 	: ', x_ref.shape)
	print('[+] Taining labels 	: ', y_ref	.shape)
	print('[+] Test set 		: ', x_test.shape)
	print('[+] Test labels 	: ', y_test.shape)
	print('[+] Validation set 	: ', x_val.shape)
	print('[+] Validation labels   : ', y_val.shape)

	return x_ref, x_test, x_val, y_ref, y_test, y_val



'''
	TFlearn simple preparation. Only transposing matrix
'''
def tf_simple_preparation(dataset):
	[x_ref, x_test, y_ref, y_test] = dataset

	print('[+] Transposing all matrixes...')
	x_ref 	= x_ref.transpose()
	x_test 	= x_test.transpose()
	y_ref 	= y_ref.transpose()
	y_test	= y_test.transpose()
	print('Done !')

	print('[+] New size of matrixes : \n x_ref : {xr} \n x_test : {xt} \n y_ref : {yr} \n y_test : {yt} \n'
		.format(xr=x_ref.shape, xt=x_test.shape, yr=y_ref.shape, yt=y_test.shape))

	return [x_ref, x_test, y_ref, y_test]






#print('\n[+] Testing file loading...')
#load_dataset_from('/home/neurones/Documents/Developpement/Dataset/_3/', 1)

#save_best_config(30)



