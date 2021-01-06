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
y_out = []
for i in range(0, N_in):
	y_in.append(x_in[i][11])
	
for i in range(0, N_out):
	y_out.append(x_out[i][11])
	
y_in = np.array(y_in)
y_out = np.array(y_out)
x_in = np.delete(x_in, -1, axis=1)
x_out = np.delete(x_out, -1, axis=1)

x_inT = x_in.transpose()

x_temp1 = x_inT.dot(x_in)
x_temp2 = np.linalg.inv(x_temp1)
pseu_invX = x_temp2.dot(x_inT)

Wlin = pseu_invX.dot(y_in.transpose())

Ebin_in = 0
Ebin_out = 0
error_in = 0
error_out = 0

for i in range(0,N_in):
	y_temp_in = 0

	for j in range(0,11):
		y_temp_in += Wlin[j]*x_in[i][j]
		
	if y_temp_in * y_in[i] <= 0:
		error_in += 1
	

for i in range(0,N_out):
	y_temp_out = 0
	
	for j in range(0,11):
		y_temp_out += Wlin[j]*x_out[i][j]
	
	if y_temp_out * y_out[i] <= 0:
		error_out += 1
		
Ebin_in = error_in/N_in
Ebin_out = error_out/N_out

print(abs(Ebin_in-Ebin_out))

