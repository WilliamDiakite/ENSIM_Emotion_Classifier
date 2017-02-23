# test numpy reshape


import numpy as np


path_directory = '/home/neurones/Documents/Developpement/Dataset/_4/'

x_ref 	= np.loadtxt(path_directory + 'CoefficientsObjet2Ref.txt')
x_test 	= np.loadtxt(path_directory + 'CoefficientsObjet2Test.txt')
y_ref 	= np.loadtxt(path_directory + 'ClasseObjet2Ref.txt')
y_test 	= np.loadtxt(path_directory + 'ClasseObjet2Test.txt')

x_ref = x_ref.T
x_test = x_test.T

print(x_ref.shape)
print(x_test.shape)


x_ref = np.reshape(x_ref, (3528,5,19,5))
print(x_ref.shape)

x_ = x_ref[1]
print(x_.shape)

print(np.reshape(x_, (1,5,19,5)).shape)

if (x_ == np.reshape(x_, (1,5,19,5))[0]).all():
	print('everythings alright bro !')

'''
x_ref 	= x_ref.transpose()
	x_ref	= reshape(x_ref, (385,13,7))
	x_test 	= x_test.transpose()
	x_test	= reshape(x_test, (150,13,7))
	
	y_ref 	= y_ref.transpose()
	y_test	= y_test.transpose()
	y_ref 	= y_ref.argmax(1)
	y_test	= y_test.argmax(1)

	y_ref = np_utils.to_categorical(y_ref, 7)
	y_test = np_utils.to_categorical(y_test, 7)	
'''