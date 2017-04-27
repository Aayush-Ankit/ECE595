import itertools #for generating permutations

num_conv = xrange(3) #no. of convolution layers
num_fc_layer = [1, 2, 3] #num FC layers
num_fc_neurons = [100, 300] #size of FC layer
feature_size_list = [10, 25, 50]

index = 0
filename = 'train_lenet_trials_var_everything.sh'

start_text = '#!/usr/bin/env sh \n\
set -e\n'

#solver_path = 'ECE595/mnist/solver/var_num_feature/'
solver_path = 'ECE595/mnist/solver/var_everything/'
#trace_path = 'ECE595/mnist/traces/var_num_feature/'
trace_path = 'ECE595/mnist/traces/var_everything/'

fid = open(filename, 'w')
fid.write(start_text)
for num_conv_layers in num_conv:
    num_feature = [p for p in itertools.product(feature_size_list, repeat=num_conv_layers)] #generates all permutations with repeatitions
    for feature_tuple in num_feature:
        for num_fcl in num_fc_layer:
            num_fcn = [q for q in itertools.product(num_fc_neurons, repeat=num_fcl)] #generates all permutations with repeatitions
            for fcn_tuple in num_fcn:
                text = './build/tools/caffe train --solver=' + solver_path + 'lenet_solver_' + str(index) + \
                    '.prototxt $@ 2>&1 | tee ' + trace_path + 'solver_' + str(index) + '_trace.txt\n'
                index = index + 1
                fid.write(text)

fid.close()
print(index)
