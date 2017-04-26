#variable kernel sizes for fixed 3 convolutional layers
num_fc_layer = [1, 2, 3] #num FC layers
filename = 'train_lenet_trials_var_fcl.sh'

start_text = '#!/usr/bin/env sh \n\
set -e\n'

solver_path = 'ECE595/mnist/solver/var_fcl/'
trace_path = 'ECE595/mnist/traces/var_fcl/'

fid = open(filename, 'w')
fid.write(start_text)
for num_fcl in num_fc_layer:
    text = './build/tools/caffe train --solver=' + solver_path + 'lenet_solver_fcl' + str(num_fcl) + \
        '.prototxt $@ 2>&1 | tee ' + trace_path + 'solver_fcl' + str(num_fcl) + '_trace.txt\n'
    fid.write(text)

fid.close()
