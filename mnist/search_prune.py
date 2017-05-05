import matplotlib.pyplot as plt
import numpy as np

def get_pareto (ds):
    # pareto optimal there does not exist a point with higher ds[i]['total_comp_cost']
    # min comp cost for each accuracy point
    index = 0
    acc = [] #list of accuracies
    y = [] #list of hardware optimal configurations
    y1 = [] #list of pareto optimal configurations
    
    for i in range(len(ds)):
        if (index == 0 or acc[index-1] < 100 * ds[i]['accuracy']): # new higher accuracy found
            acc.append(ds[i]['accuracy'] * 100)
            y.append(ds[i])
            index = index + 1 # next new accuracy index
        elif (ds[i]['total_comp_cost'] < y[index-1]['total_comp_cost']): #lower cost ption found dor same accuracy
            y[index-1] = ds[i]

    # check pareto dominated for every element
    # for every element with unique accuracy and minimum hardware cost check if any configuration exists for higher accuracy with lower hardware cost        
    for i in range(len(y)): 
        pareto_dominated = 0
        for j in range(i,len(y)):
             if (y[i]['accuracy'] <= y[j]['accuracy']  and y[i]['total_comp_cost'] > y[j]['total_comp_cost']):
                 pareto_dominated = 1
                 break
        if (pareto_dominated != 1):
            y1.append(y[i])
    return y1

# Original search spaces
d = np.load('net_acc2.npy')
d1 = np.load('net_acc.npy')
dnet = np.append(d, d1)

ds = sorted(dnet, key=lambda k:k ['accuracy'])

#prune the search space (feature map fan-out)
dsp = []
for dict_t in ds:
    feature_tuple = dict_t['feature_list']
    if (tuple(sorted(feature_tuple)) == feature_tuple): #num features should increase across layers
        dsp.append(dict_t)

# extract pareto optimal front of original and pruned search space
ds_pareto = get_pareto(ds)
dsp_pareto = get_pareto(dsp)

prune_stats = {'original':len(ds), 'original_pareto':len(ds_pareto),
         'pruned':len(dsp), 'pruned_pareto':len(dsp_pareto)}
np.save('prune_results.npy', prune_stats)
