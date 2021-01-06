import numpy as np
import random

x = []
with open('hw1_train.dat') as f1:
	line = f1.readlines()
	for l in line:
		x.append(np.fromstring("1.00000	"+l, dtype=float, sep='	'))

w = np.zeros(11)


N = len(x)
repeatTime = 1000
updateTimes = []
w0 = []

for repeat in range(0, repeatTime):
	updates = 0
	w = np.zeros(11)
	i = 0
	random.seed(repeat)
	while rand != N:
		rand = random.randint(0,N-1)
		sum = 0
		result = float(0)
		
		for j in range(0,11):
				sum += x[rand][j]*w[j]
		
		if sum > 0:
			result = 1
		else:
			result = -1
	
		if result*x[rand][11] != 1:
			updates += 1
			i = -1
			rand = 0
			for k in range(0,11):
				w[k] += x[rand][k]*x[rand][11]
		
		i += 1
		rand += 1
	
	updateTimes.append(updates)
	w0.append(w[0])
#with open('hw1_result.dat', 'w') as f2:
#	for ww in w:
#		f2.write(str(ww) + "	")

print(np.median(updates))
