import numpy as np
import random
from numpy.linalg import *
import statistics
import math
x_in = []
with open('hw3_train.dat') as f1:
	line = f1.readlines()
	for l in line:
		x_in.append(np.fromstring("1.00000	"+l, dtype=float, sep='	'))


x_out = []
with open('hw3_test.dat') as f1:
	line = f1.readlines()
	for l in line:
		x_out.append(np.fromstring("1.00000	"+l, dtype=float, sep='	'))
		

N_in = len(x_in)
N_out = len(x_out)

y_in = []
for i in range(0, N_in):
	y_in.append(x_in[i][11])

y_in = np.array(y_in)
x_in = np.delete(x_in, -1, axis=1)

repeatTime = 1000
updates = []
eta = 0.001

for repeat in range(0, repeatTime):
	random.seed(repeat)
	w = np.zeros(11)
	error_SGD = 0
	

	for iteration in range(0, 500):
		rand = random.randint(0,N_in-1)
		y_temp_SGD = 0
		s = 0
		for j in range(0,11):
			s += -1*w[j]*x_in[rand][j]*y_in[rand]
		thetaS = 1/(1+math.exp(-s))
		for j in range(0,11):
			w[j] += eta*thetaS*y_in[rand]*x_in[rand][j]
		
	CE_in_SGD = 0
	for k in range(0,N_in):
		y_temp_SGD = 0
		for j in range(0,11):
			y_temp_SGD += y_in[k]*w[j]*x_in[k][j]
		CE_in_SGD += np.log(1+math.exp(-y_temp_SGD))
		
	CE_in_SGD /= N_in

	updates.append(CE_in_SGD)
	
print(statistics.mean(updates))