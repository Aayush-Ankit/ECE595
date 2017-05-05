import matplotlib.pyplot as plt
import numpy as np

#pareto optimal there does not exist a point with higher ds[i]['total_comp_cost']
d = np.load('net_acc2.npy')
d1 = np.load('net_acc.npy')
dnet = np.append(d, d1)

ds = sorted(dnet, key=lambda k:k ['accuracy'])

acc = [] #list of accuracies
acc2 = [] #list of pareto optimal accuracies
y = [] #list of hardware optimal configurations
y1 = [] #list of pareto optimal configurations
y2 = [] #list of number of convolutional layers for pareto optimal configurations
y3 = [] # list of cost for pareto optimal configuration
num_conv = [] # list of number of convolutional layers for pareto optimal
num_fcl = [] # list of number of fully connected neurons
num_features0 = []
num_features1 = []

index = 0

for i in range(len(ds)):
    if (index == 0 or acc[index-1] < 100 * ds[i]['accuracy']): # new higher accuracy found
        acc.append(ds[i]['accuracy'] * 100)
        y.append(ds[i])
        index = index + 1 # next new accuracy index
    elif (ds[i]['total_comp_cost'] < y[index-1]['total_comp_cost']): #lower cost ption found dor same accuracy
        y[index-1] = ds[i]

#check pareto dominated for every element
#for every element with unique accuracy and minimum hardware cost check if any configuration exists for higher accuracy with lower hardware cost        
for i in range(len(y)): 
    pareto_dominated = 0
    for j in range(i,len(y)):
         if (y[i]['accuracy'] <= y[j]['accuracy']  and y[i]['total_comp_cost'] > y[j]['total_comp_cost']):
             pareto_dominated = 1
             break
    if (pareto_dominated != 1):
        y1.append(y[i])
        acc2.append(y[i]['accuracy'] * 100)
        y3.append(y[i]['total_comp_cost'])

        num_fcl.append(y[i]['num_fcl'])
        num_conv.append(y[i]['num_conv'])
        if (len(y[i]['feature_list']) > 0):
            num_features0.append(y[i]['feature_list'][0])
        else:
            num_features0.append(0)
        if (len(y[i]['feature_list']) > 1):
            num_features1.append(y[i]['feature_list'][1])
        else:
            num_features1.append(0)
        
np.save('pareto_optimal.npy', y1)

plt.scatter(acc2, y3, color='red', label = 'pareto optimal total_comp_cost')
plt.ylim([0,200000])
plt.xlim([90,100])
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("Computational cost")
plt.title ("pareto optimal computational cost versus Accuracy")
#plt.show()
plt.savefig('pareto_optimal.png')

print("Original number of elements", len(ds), " hardware optimal points with unique accuracies ", len(y), " pareto optimal elements ", len(y1))

plt.figure()
plt.scatter(acc2, num_conv, label = 'num_conv')
plt.ylim([0,3])
plt.xlim([90,100])
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("number of convolutional layers")
plt.title ("pareto optimal number of convolutional layers versus Accuracy")
#plt.show()
plt.savefig('pareto_optimal_numconv.png')

plt.figure()
plt.scatter(acc2, num_fcl, label = 'num_fcl')
plt.ylim([0,4])
plt.xlim([90,100])
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("number of fully connected layers")
plt.title ("pareto optimal number of fully connected layers versus Accuracy")
#plt.show()
plt.savefig('pareto_optimal_num_fcl.png')

plt.figure()
plt.scatter(acc2, num_features0, label = 'num_features[0]')
plt.ylim([0,100])
plt.xlim([90,100])
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("number of features in conv[0]")
plt.title ("pareto optimal number of features in conv[0] versus Accuracy")
#plt.show()
plt.savefig('pareto_optimal_num_features0.png')

plt.figure()
plt.scatter(acc2, num_features1, label = 'num_features[1]')
plt.ylim([0,100])
plt.xlim([90,100])
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("number of features in conv[1]")
plt.title ("pareto optimal number of features in conv[1] versus Accuracy")
#plt.show()
plt.savefig('pareto_optimal_num_features1.png')

