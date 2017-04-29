import matplotlib.pyplot as plt
import numpy as np

d = np.load('net_acc2.npy')

x = []
y = []
y2 = []
y3 = []
max_Acc = 0 #maximum accuracy
max_cost = 0
min_Acc = 10 #minimum accuracy
min_cost = 1000000

for i in range(len(d)):
    if (d[i]['accuracy'] > float(max_Acc)):
        max_Acc = d[i]['accuracy']
        max_Acc_idx = i
    if (d[i]['total_comp_cost'] > float(max_cost)):
        max_cost = d[i]['total_comp_cost']
        max_cost_idx = i
    if (d[i]['accuracy'] < float(min_Acc)):
        min_Acc = d[i]['accuracy']
        min_Acc_idx = i
    if (d[i]['total_comp_cost'] < float(min_cost)):
        min_cost = d[i]['total_comp_cost']
        min_cost_idx = i
    x.append(d[i]['accuracy'] * 100)
    y.append(d[i]['total_comp_cost'])
    y2.append(d[i]['fcl_cost'])
    y3.append(d[i]['conv_cost'])

print("Maximum accuracy is %f for index %d" % (max_Acc, max_Acc_idx))
print("Maximum cost is %f for index %d" % (max_cost, max_cost_idx))
print("Maximum accuracy is %f for index %d" % (min_Acc, min_Acc_idx))
print("Maximum cost is %f for index %d \n" % (min_cost, min_cost_idx))

plt.scatter(x,y, color='red', label = 'total_comp_cost')
plt.scatter(x,y2, color='green', label = 'fcl_cost')
plt.scatter(x,y3, label = 'conv_cost')
plt.ylim([0,800000])
plt.xlim([40,100])
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("Computational cost")
plt.title ("Computational cost versus Accuracy")
#plt.show()
plt.savefig('compute_accuracy2.png')
             
