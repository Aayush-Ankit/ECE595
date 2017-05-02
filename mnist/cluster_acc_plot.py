
import matplotlib.pyplot as plt
import numpy as np

d = np.load('net_acc2.npy')
d1 = np.load('net_acc.npy')

x = []
y = []
y2 = []
y3 = [] # number of features in first conv layer
x3 = []
y4 = [] # number of features in 2nd conv layer
x4 = []
y5 = [] # number of neurons in the 1st fully connected layer
x5 = []
y6 = [] # number of neurons in the 2nd fully connected layer
x6 = []
y7 = [] # number of neurons in the 3rd fully connected layer
x7 = []

for i in range(len(d)):
    if ((d[i]['accuracy'] > 0.6) &  (d[i]['accuracy'] < 0.7)):
        x.append(d[i]['accuracy'] * 100)
        y.append(d[i]['num_fcl'])
        y2.append(d[i]['num_conv'])
        if (len(d[i]['feature_list']) > 0):
            y3.append(d[i]['feature_list'][0])
            x3.append(d[i]['accuracy'] * 100)
        if (len(d[i]['feature_list']) > 1):
            y4.append(d[i]['feature_list'][1])
            x4.append(d[i]['accuracy'] * 100)
        y5.append(d[i]['num_fcn'][0])
        if (len(d[i]['num_fcn']) > 1):
            x6.append(d[i]['accuracy'] * 100)
            y6.append(d[i]['num_fcn'][1])
        if (len(d[i]['num_fcn']) > 2):
            y7.append(d[i]['num_fcn'][2])
            x7.append(d[i]['accuracy'] * 100)

for i in range(len(d1)):
    if ((d1[i]['accuracy'] > 0.6) &  (d1[i]['accuracy'] < 0.7)):
        x.append(d1[i]['accuracy'] * 100)
        y.append(d1[i]['num_fcl'])
        y2.append(d1[i]['num_conv'])
        if (len(d1[i]['feature_list']) > 0):
            y3.append(d1[i]['feature_list'][0])
            x3.append(d[i]['accuracy'] * 100)
        if (len(d1[i]['feature_list']) > 1):
            y4.append(d1[i]['feature_list'][1])
            x4.append(d[i]['accuracy'] * 100)
        y5.append(d1[i]['num_fcn'][0])
        if (len(d1[i]['num_fcn']) > 1):
            x6.append(d[i]['accuracy'] * 100)
            y6.append(d1[i]['num_fcn'][1])
        if (len(d1[i]['num_fcn']) > 2):
            y7.append(d1[i]['num_fcn'][2])
            x7.append(d[i]['accuracy'] * 100)

plt.figure(0)
plt.scatter(x,y, color='red', label = 'num_fcl')
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("Number of fully connected layers")
plt.xlim([60,70])
plt.savefig('num_fcl_accuracy_comb.png')

#new figure
plt.figure(1)
plt.scatter(x,y2, label = 'num_conv')
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("Number of convolutional layers")
plt.xlim([60,70])
plt.savefig('num_conv_accuracy_comb.png')
#plt.show()

#new figure
fig1 = plt.figure(2)
#ax1 = fig1.add_subplot(111)
#fig1.subplots_adjust(hspace=.5)
#ax1.scatter(x3,y3, label = 'num_feature conv[0]')
#ax1.set_xlabel("Accuracy (%)")
#ax1.set_ylabel("Number of features[0]")
#ax1.set_xlim([60,70])
plt.scatter(x3,y3, label = 'num_feature conv[0]')
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("Number of features[0]")
plt.xlim([60,70])
plt.savefig('num_features[0]_accuracy_comb.png')

fig2 = plt.figure(3)     
##ax2 = fig2.add_subplot(111)
##ax2.scatter(x4,y4, label = 'num_feature conv[1]')
##ax2.set_xlabel("Accuracy (%)")
##ax2.set_ylabel("Number of features[1]")
##ax2.set_xlim([60,70])
plt.scatter(x4,y4, label = 'num_feature conv[0]')
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("Number of features[1]")
plt.xlim([60,70])
plt.savefig('num_features[1]_accuracy_comb.png')

plt.figure(4)     
plt.scatter(x,y5, label = 'num_fcn[0]')
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("Number of fcn[0]")
plt.xlim([60,70])
plt.savefig('num_fcn[0]_accuracy_comb.png')

plt.figure(5)     
plt.scatter(x6,y6, label = 'num_fcn[1]')
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("Number of fcn[1]")
plt.xlim([60,70])
plt.savefig('num_fcn[1]_accuracy_comb.png')

plt.figure(6)     
plt.scatter(x7,y7, label = 'num_fcn[2]')
plt.xlabel("Accuracy (%)")
plt.legend(loc = 'upper left')
plt.ylabel("Number of fcn[1]")
plt.xlim([60,70])
plt.savefig('num_fcn[2]_accuracy_comb.png')
             
