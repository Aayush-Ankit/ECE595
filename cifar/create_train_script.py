#variable kernel sizes for fixed 3 convolutional layers
feature_size_list = [10, 25, 50, 100, 150]
filename = 'train_cifarlenet_trials_var_featuresize.sh'

start_text = '#!/usr/bin/env sh \n\
set -e\n'

solver_path = 'ECE595/cifar/solver/var_num_feature_maxiter20000/'
trace_path = 'ECE595/cifar/traces/var_num_feature_maxiter20000/'

fid = open(filename, 'w')
fid.write(start_text)
for f_size in feature_size_list:
    text = './build/tools/caffe train --solver=' + solver_path + 'cifar_solver_featuresize' + str(f_size) + \
        '.prototxt $@ 2>&1 | tee ' + trace_path + 'solver_featuresize' + str(f_size) + '_trace.txt\n'
    fid.write(text)

fid.close()



