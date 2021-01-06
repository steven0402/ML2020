import numpy as np
import random
from numpy.linalg import *
import statistics

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

x_inT = x_in.transpose()

x_temp1 = x_inT.dot(x_in)
x_temp2 = np.linalg.inv(x_temp1)
pseu_invX = x_temp2.dot(x_inT)

Wlin = pseu_invX.dot(y_in.transpose())

Esqr_in = 0
error = 0
for i in range(0,N_in):
	y_temp = 0
	for j in range(0,11):
		y_temp += Wlin[j]*x_in[i][j]
	error += ((y_temp - y_in[i]) ** 2)
	
Esqr_in_wlin = error/N_in



repeatTime = 1000
updates = []
eta = 0.001

for repeat in range(0, repeatTime):
	random.seed(repeat)
	iteration = 0
	w = np.zeros(11)
	Esqur_in_SGD = 1.01*Esqr_in_wlin + 1
	error_SGD = 0

	while Esqur_in_SGD > 1.01*Esqr_in_wlin:
		rand = random.randint(0,N_in-1)
		y_temp_SGD = 0
		
		wx = 0
		for j in range(0,11):
			wx += w[j]*x_in[rand][j]
			
		for j in range(0,11):
			w[j] += eta*2*(y_in[rand] - wx)*x_in[rand][j]
		
		Esqur_in_SGD = 0
		for k in range(0,N_in):
			y_temp_SGD = 0
			for j in range(0,11):
				y_temp_SGD += w[j]*x_in[k][j]
			Esqur_in_SGD +=((y_temp_SGD - y_in[k]) ** 2)
		
		Esqur_in_SGD /= N_in

		iteration += 1
	print(str(repeat) + ' ' + str(iteration))
	updates.append(iteration)
	
print(statistics.mean(updates))