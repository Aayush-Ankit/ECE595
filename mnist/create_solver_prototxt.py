num_trials = 10
path = "ECE595/mnist/train_test/var_ker_size/"
solver_path = "solver/var_ker_size/"

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
max_iter: 10000\n\
# snapshot intermediate results\n\
snapshot: 5000\n\
snapshot_prefix: \"examples/mnist/lenet\"\n\
# solver mode: CPU or GPU\n\
solver_mode: CPU"

num_layers = 3
kernel_size_list = [3, 5]

#for i in xrange(num_trials):
for k_size in kernel_size_list:
    filename = solver_path + 'lenet_solver_kersize' + str(k_size) + '.prototxt'
    fid = open(filename, 'w')
    fid.write("# The train/test net protocol buffer definition\n")
    network_filename = "net: \"" + path + 'lenet_train_test_kersize' + str(k_size) + ".prototxt\"\n"
    fid.write(network_filename)
    fid.write(common_text)
    fid.close()
