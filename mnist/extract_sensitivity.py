#Analyze computation cost vs number of pruned datapoints 
import matplotlib.pyplot as plt
import numpy as np

#Load the explored datapoints
d = np.load('net_acc2.npy')
d1 = np.load('net_acc.npy')
dnet = np.append(d, d1)

#Extract the computational cost values from dnet in sorted form
comp_cost = []
for i in range(len(dnet)):
    temp = dnet[i]['total_comp_cost']
    comp_cost = np.append(comp_cost, temp)
comp_cost = sorted(comp_cost)

#Collect data and plot the sensitivity list
num_points = 1000
min_comp_cost = min(comp_cost)
max_comp_cost = max(comp_cost)
stepsize = (max_comp_cost - min_comp_cost + 1) / num_points
y = []
x = []
for i in range(num_points):
    x = np.append(x, min_comp_cost + i*stepsize)
    y_temp = len(comp_cost) - sum(comp > x[i] for comp in comp_cost)
    y = np.append(y, y_temp)

plt.plot(x,y, color = 'red', label = 'sensitivity with computation cost')
plt.xlabel("Computation Cost")
plt.legend(loc = 'upper left')
plt.ylabel("Remaining search space")
plt.title ("sensitivity with computation cost")
plt.savefig('sensitivity.png')
