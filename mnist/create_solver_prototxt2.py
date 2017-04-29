num_trials = 10
import itertools #for generating permutations

#path = "ECE595/mnist/train_test/var_ker_size/"
#solver_path = "solver/var_ker_size/"

path = 'ECE595/mnist/train_test/var_everything2/'
#solver_path = "solver/var_num_feature/"
solver_path = "solver/var_everything2/"

# 1000 iterations --> 1 epoch (64*1000=64000 MNIST images)
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
max_iter: 1000\n\
# snapshot intermediate results\n\
snapshot: 5000\n\
snapshot_prefix: \"examples/mnist/lenet\"\n\
# solver mode: CPU or GPU\n\
solver_mode: GPU"

num_conv = xrange(3) #no. of convolution layers
num_fc_layer = [1, 2, 3] #num FC layers
num_fc_neurons = [10, 25, 50] #size of FC layer

index = 0

feature_size_list = [2, 5, 10, 25, 50, 100]
for num_conv_layers in num_conv:
    num_feature = [p for p in itertools.product(feature_size_list, repeat=num_conv_layers)] #generates all permutations with repeatitions
    for feature_tuple in num_feature:
        for num_fcl in num_fc_layer:
            num_fcn = [q for q in itertools.product(num_fc_neurons, repeat=num_fcl)] #generates all permutations with repeatitions
            for fcn_tuple in num_fcn:
                filename = solver_path + 'lenet_solver_' + str(index) + '.prototxt'
                fid = open(filename, 'w')
                fid.write("# The train/test net protocol buffer definition\n")
                network_filename = "net: \"" + path + 'lenet_train_test' + str(index) + ".prototxt\"\n"
                index = index + 1
                fid.write(network_filename)
                fid.write(common_text)
                fid.close()


print(index)
