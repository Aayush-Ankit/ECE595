import itertools

# Using CIFAR10 training solver from examples - cifar10_full_sigmoid_solver.prototxt
path = "ECE595/cifar/train_test/var_everything_tiny/"
solver_path = "./solver/var_everything_tiny/"

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
max_iter: 60000\n\
# snapshot intermediate results\n\
snapshot: 10000\n\
snapshot_prefix: \"examples/cifar10_full_sigmoid\"\n\
# solver mode: CPU or GPU\n\
solver_mode: GPU"

# Code to generate prototxt files
ker_size_list = [3,5] #kernel sizes
num_conv = [1,2,3] #number of conv layers
num_fc_layer = [1,2]
num_fc_neurons = [10,60,180]
feature_size_list = [24, 72, 216]

index = 0
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
                    filename = solver_path + 'cifar_solver_' + str(index) + '.prototxt'
                    fid = open(filename, 'w')
                    fid.write("# reduce learning rate after 120 epochs (60000 iters) by factor 0f 10\n\
# then another factor of 10 after 10 more epochs (5000 iters)\n\n")
                    fid.write("# The train/test net protocol buffer definition\n")
                    network_filename = "net: \"" + path + 'cifar_train_test_' + str(index) + ".prototxt\"\n"
                    index = index + 1
                    fid.write(network_filename)
                    fid.write(common_text)
                    fid.close()

print (index)
