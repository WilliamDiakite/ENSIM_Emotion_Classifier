# Test open all sample

import scipy
from scipy import io

import numpy
from numpy import reshape
from PIL import Image

from random import shuffle

import os
from sys import exit


directory = '/home/neurones/Documents/Developpement/Dataset/_3/'

i = 0
y = numpy.zeros((11, 1))
x = numpy.zeros((len(os.listdir(directory)), 5, 19))

print('y intial shape : ', y.shape)

'''
for filename in os.listdir(directory):
	if filename.endswith(".mat"):
		full_name = directory + '/' + filename
		current_sample = scipy.io.loadmat(full_name)
		x[i] = reshape(numpy.asarray(current_sample['Vec']), (5,19))
		y = numpy.concatenate((y, numpy.asarray(current_sample['Class'])), axis=1)
		i += 1
'''

maList = os.listdir(directory)[:]
shuffle(maList)

print('ma liste : ', len(maList))

#print('Class : ', current_sample['Class'].shape)
#print('Data : ', current_sample['Vec'].shape)
#print(current_sample['Class'])

#y = numpy.delete(y, 0, 1)
#y = y.T

#print(x.shape)
#print(y.shape)


