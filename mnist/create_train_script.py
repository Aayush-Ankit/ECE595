#variable kernel sizes for fixed 3 convolutional layers
kernel_size_list = [3, 5]
filename = 'train_lenet_trials_var_ker_size.sh'

start_text = '#!/usr/bin/env sh \n\
set -e\n'

solver_path = 'ECE595/mnist/solver/var_ker_size/'
trace_path = 'ECE595/mnist/traces/var_ker_size/'

fid = open(filename, 'w')
fid.write(start_text)
for k_size in kernel_size_list:
    text = './build/tools/caffe train --solver=' + solver_path + 'lenet_solver_kersize' + str(k_size) + \
            '.prototxt $@ 2>&1 | tee ' + trace_path + 'solver_kersize' + str(k_size) + '_trace.txt\n'
    fid.write(text)
    
fid.close()


    
