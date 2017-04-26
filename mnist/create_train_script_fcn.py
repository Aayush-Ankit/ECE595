#variable kernel sizes for fixed 3 convolutional layers
num_fc_neurons = [100, 300, 500] #size of FC layer
filename = 'train_lenet_trials_var_fcn.sh'

start_text = '#!/usr/bin/env sh \n\
set -e\n'

solver_path = 'ECE595/mnist/solver/var_fcn/'
trace_path = 'ECE595/mnist/traces/var_fcn/'

fid = open(filename, 'w')
fid.write(start_text)
for num_fcn in num_fc_neurons:
    text = './build/tools/caffe train --solver=' + solver_path + 'lenet_solver' + '_fcn' + str(num_fcn) + \
        '.prototxt $@ 2>&1 | tee ' + trace_path + 'solver' + '_fcn' + str(num_fcn) + '_trace.txt\n'
    fid.write(text)

fid.close()


    
