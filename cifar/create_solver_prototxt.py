# Using CIFAR10 training solver from examples - cifar10_full_sigmoid_solver.prototxt
path = "ECE595/cifar/train_test/var_num_feature_maxiter6000/"
solver_path = "./solver/var_num_feature_maxiter6000_lr0.01/"

# 10000 iterations --> 10 epoch (64*1000=64000 CIFAR images)
# writing the solver prototxt
common_text = "# test_iter specifies how many forward passes the test should carry out.\n\
# In the case of CIFAR10, we have test batch size 100 and 100 test iterations,\n\
# covering the full 10,000 testing images.\n\
test_iter: 10\n\
# Carry out testing every 1000 training iterations.\n\
test_interval: 1000\n\
# The base learning rate, momentum and the weight decay of the network.\n\
base_lr: 0.001\n\
momentum: 0.9\n\
#weight_decay: 0.004\n\
# The learning rate policy\n\
lr_policy: \"step\"\n\
gamma: 1\n\
stepsize: 5000\n\
# Display every 100 iterations\n\
display: 100\n\
# The maximum number of iterations\n\
max_iter: 6000\n\
# snapshot intermediate results\n\
snapshot: 10000\n\
snapshot_prefix: \"examples/cifar10_full_sigmoid\"\n\
# solver mode: CPU or GPU\n\
solver_mode: GPU"

feature_size_list = [10, 25, 50, 100, 150]
for f_size in feature_size_list:
    filename = solver_path + 'cifarlenet_solver_featuresize' + str(f_size) + '.prototxt'
    fid = open(filename, 'w')
    fid.write("# reduce learning rate after 120 epochs (60000 iters) by factor 0f 10\n\
# then another factor of 10 after 10 more epochs (5000 iters)\n\n")
    fid.write("# The train/test net protocol buffer definition\n")
    network_filename = "net: \"" + path + 'cifarlenet_train_test_featuresize' + str(f_size) + ".prototxt\"\n"
    fid.write(network_filename)
    fid.write(common_text)
    fid.close()





