import numpy as np
import random
from numpy.linalg import *

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

#X^T*X
x_temp1 = x_inT.dot(x_in)
#(X^T*X)^-1
x_temp2 = np.linalg.inv(x_temp1)
#(X^T*X)^-1*X^T
pseu_invX = x_temp2.dot(x_inT)
#(X^T*X)^-1*X^Ty
Wlin = pseu_invX.dot(y_in.transpose())

Esqr_in = 0
error = 0
for i in range(0,N_in):
	y_temp = 0
	for j in range(0,11):
		y_temp += Wlin[j]*x_in[i][j]
	error += ((y_temp - y_in[i]) ** 2)
	
Esqr_in = error/N_in

print(Esqr_in)

