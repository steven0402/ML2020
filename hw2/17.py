import numpy as np
import random
import statistics

tau = 0
N = 20
repeat_time = 10000
s = [-1, 1]
diff_list = []
sample_size = 100000

for re in range(0, repeat_time):
	i = 0
	x1 = []
	exist = []
	while i < N:
		randn = random.uniform(-1, 1)
		if randn not in exist:
			exist.append(randn)
			
			change_sign = random.uniform(0, 1)
			sign = 1
			if change_sign < tau:
				sign = -1
			
			if randn > 0:
				y = 1*sign
			else:
				y = -1*sign
				
			x1.append((randn, y))
			i = i + 1
				
	x1.sort(key=lambda x:x[0])
	theta = -1
	bestTheta = -1
	bestEin = 1
	bestS = -1
	for s1 in s:
		Ein = 0
		for point in x1:
			if(point[0]-theta) > 0:
				sign = 1*s1
			else:
				sign = -1*s1
			if sign != point[1]:
				Ein = Ein + 1
			
		if Ein/N < bestEin:
			bestEin = Ein/N
			bestS = s1
			bestTheta = -1
		
	for i in range(0, N-1):
		theta = (x1[i][0]+x1[i+1][0])/2
		for s1 in s:
			Ein = 0
			for point in x1:
				if(point[0]-theta) > 0:
					sign = 1*s1
				else:
					sign = -1*s1
				if sign != point[1]:
					Ein = Ein + 1

			if Ein/N < bestEin:
				bestEin = Ein/N
				bestS = s1
				bestTheta = theta
					
	
	Eout = 0.5*abs(bestTheta)*(1-tau) + (1-0.5*abs(bestTheta))*tau
		
	diff_list.append(Eout-bestEin)
	
print(statistics.mean(diff_list))