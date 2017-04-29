import numpy as np
import itertools #for generating permutations

# CNN configurations
num_conv = xrange(3) #no. of convolution layers
num_fc_layer = [1, 2, 3] #num FC layers
num_fc_neurons = [10, 25, 50] #size of FC layer
feature_size_list = [2, 5, 10, 25, 50, 100]

# List of dictionaries to store CNN configs and accuracy
acc_list = []

# Path to trace
path = './traces/var_everything2/solver_'

# Relative cost for MAC, MEMORY, ADD
cost_mac = 1;
cost_mem = 2;
#cost_compute = 1;

# Other parameters
img_size = 32
num_inp_dim = 1
num_classes = 10
ker_size = 5
maxpool_size = 2

index = 0
for num_conv_layers in num_conv:
    num_feature = [p for p in itertools.product(feature_size_list, repeat=num_conv_layers)] #generates all permutations with repeatitions
    for feature_tuple in num_feature:
        for num_fcl in num_fc_layer:
            num_fcn = [q for q in itertools.product(num_fc_neurons, repeat=num_fcl)] #generates all permutations with repeatitions
            for fcn_tuple in num_fcn:
                
                #create dictionary
                temp_dict = {}
                temp_dict['idx'] = index
                temp_dict['num_conv'] = num_conv_layers
                temp_dict['num_fcl'] = num_fcl
                temp_dict['feature_list'] = feature_tuple
                temp_dict['num_fcn'] = fcn_tuple

                #Open the corresponding trace file to extract accuracy
                filename = path + str(index) + '_trace.txt'
                with open (filename) as file:
                    last_line = file.readlines()[-4]
                start = str.find(last_line, '=') + 2
                end = str.find(last_line, '\n')

                temp_dict['accuracy'] = float(last_line[start:end])
                

                # Compute convolution layers computation cost
                temp_dict['conv_cost'] = 0
                inp_size = img_size
                num_in_map = num_inp_dim
                for j in xrange(num_conv_layers):
                    num_out_map = feature_tuple[j] #index starts at 0 for tuple
                    out_size = inp_size - ker_size + 1
                    inp_size = out_size / maxpool_size #after the max-pool (for next conv layer)
                    temp_mac = (num_out_map * (out_size^2))* (num_in_map * (ker_size^2))
                    temp_mem = num_out_map*num_in_map*(ker_size^2)
                    temp_dict['conv_cost'] = temp_dict['conv_cost'] + temp_mem*cost_mem + temp_mac*cost_mac
                    num_in_map = num_out_map
                

                # Compute FC layers computation cost
                temp_dict['fcl_cost'] = 0
                for j in xrange(num_fcl):
                    out_size = fcn_tuple[j]
                    temp_mem = out_size * inp_size
                    temp_mac = temp_mem
                    inp_size = out_size
                    temp_dict['fcl_cost'] = temp_dict['fcl_cost'] + temp_mem*cost_mem + temp_mac*cost_mac

                # Total CNN cost
                temp_dict['total_comp_cost'] = temp_dict['conv_cost'] + temp_dict['fcl_cost']
                
                # append values to list
                acc_list.append(temp_dict)
                index = index + 1
                
print(index)
np.save('net_acc2.npy', acc_list)
