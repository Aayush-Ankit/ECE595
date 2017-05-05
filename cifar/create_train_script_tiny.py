import itertools

solver_batch_size = 200

ker_size_list = [3,5] #kernel sizes
num_conv = [1,2,3] #number of conv layers
num_fc_layer = [1,2]
num_fc_neurons = [10,60,180]
feature_size_list = [24, 72, 216]

index = 0

start_text = '#!/usr/bin/env sh \n\
set -e\n'

solver_path = 'ECE595/cifar/solver/var_everything_tiny/'
trace_path = 'ECE595/cifar/traces/var_everything_tiny/'
filename = './trainfiles/var_everything_tiny/train_cifarlenet_trials_var_everything.sh' #opening a dummy file -just to close
fid = open(filename, 'w')
fid.write(start_text)

solver_current = 0
batch_num = 0
for num_conv_layers in num_conv:
    num_feature = [p for p in itertools.product(feature_size_list, repeat=num_conv_layers)] #generate all permutations with repetitions
    ker_size = [r for r in itertools.product(ker_size_list, repeat=num_conv_layers)] #generate all permutations with repetitions
    for feature_tuple in num_feature: #choose feature
        if (tuple(sorted(feature_tuple)) != feature_tuple): #num features should increase across layers
            continue
        for kernel_tuple in ker_size: #choose kernel
            if ((tuple(sorted(kernel_tuple, reverse=True)) != kernel_tuple) or (kernel_tuple == (5,5,5))): #skip impossible configs
                continue
            for num_fcl in num_fc_layer:
                num_fcn = [q for q in itertools.product(num_fc_neurons, repeat=num_fcl)] #generate all permutations
                for fcn_tuple in num_fcn: #chose fc neurons

                    # write a batch of solvers on a file
                    if (solver_current % solver_batch_size == 0):
                        fid.close()
                        batch_num = batch_num + 1
                        filename = './trainfiles/var_everything_tiny/train_cifarlenet_trials_var_everything_' + str(batch_num) + '.sh'
                        fid = open(filename, 'w')

                    solver_current = solver_current + 1
                    text = './build/tools/caffe train --solver=' + solver_path + 'cifar_solver_' + str(index) + \
                        '.prototxt $@ 2>&1 | tee ' + trace_path + 'solver_' + str(index) + '_trace.txt\n'
                    index = index + 1
                    fid.write(text)

fid.close()
print (index)
print(batch_num)



