num_trials = 10
#path = "ECE595/mnist/train_test/var_ker_size/"
#solver_path = "solver/var_ker_size/"

path = "ECE595/mnist/train_test/var_num_feature/"
#solver_path = "solver/var_num_feature/"
solver_path = "solver/var_num_feature_maxiter100/"

# writing the solver prototxt
common_text = "# test_iter specifies how many forward passes the test should carry out.\n\
# In the case of MNIST, we have test batch size 100 and 100 test iterations,\n\
# covering the full 10,000 testing images.\n\
test_iter: 100\n\
# Carry out testing every 500 training iterations.\n\
test_interval: 500\n\
# The base learning rate, momentum and the weight decay of the network.\n\
base_lr: 0.01\n\
momentum: 0.9\n\
weight_decay: 0.0005\n\
# The learning rate policy\n\
lr_policy: \"inv\"\n\
gamma: 0.0001\n\
power: 0.75\n\
# Display every 100 iterations\n\
display: 100\n\
# The maximum number of iterations\n\
max_iter: 100\n\
# snapshot intermediate results\n\
snapshot: 5000\n\
snapshot_prefix: \"examples/mnist/lenet\"\n\
# solver mode: CPU or GPU\n\
solver_mode: CPU"

##num_layers = 3
##kernel_size_list = [3, 5]

##num_fc_layer = [1, 2, 3] #num FC layers
##num_fc_neurons = [100, 300, 500] #size of FC layer
##for num_fcl in num_fc_layer:
##    for num_fcn in num_fc_neurons:
##        filename = solver_path + 'lenet_solver_fcl' + str(num_fcl) + '_fcn' + str(num_fcn) + '.prototxt'
##        fid = open(filename, 'w')
##        fid.write("# The train/test net protocol buffer definition\n")
##        network_filename = "net: \"" + path + 'lenet_train_test_fcl' + str(num_fcl) + '_fcn' + str(num_fcn) + '.prototxt\"\n'
##        fid.write(network_filename)
##        fid.write(common_text)
##        fid.close()

feature_size_list = [10, 25, 50, 100, 150]
for f_size in feature_size_list:
    filename = solver_path + 'lenet_solver_featuresize' + str(f_size) + '.prototxt'
    fid = open(filename, 'w')
    fid.write("# The train/test net protocol buffer definition\n")
    network_filename = "net: \"" + path + 'lenet_train_test_featuresize' + str(f_size) + ".prototxt\"\n"
    fid.write(network_filename)
    fid.write(common_text)
    fid.close()





