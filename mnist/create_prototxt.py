num_trials = 10

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

for i in xrange(num_trials):
    filename = 'lenet_solver' + str(i) + '.prototxt'
    fid = open(filename, 'w')
    fid.write("# The train/test net protocol buffer definition\n")
    network_filename = "net: \"ECE595/mnist/train_test/lenet_train_test" + str(i) + ".prototxt\"\n"
    fid.write(network_filename)
    fid.write(common_text)
    fid.close()

# writing the train_test prototxt
common_text_start = "name: \"LeNet\"\n\
layer {\n\
  name: \"mnist\"\n\
  type: \"Data\"\n\
  top: \"data\"\n\
  top: \"label\"\n\
  include {\n\
    phase: TRAIN\n\
  }\n\
  transform_param {\n\
    scale: 0.00390625\n\
  }\n\
  data_param {\n\
    source: \"examples/mnist/mnist_train_lmdb\"\n\
    batch_size: 64\n\
    backend: LMDB\n\
  }\n\
}\n\
layer {\n\
  name: \"mnist\"\n\
  type: \"Data\"\n\
  top: \"data\"\n\
  top: \"label\"\n\
  include {\n\
    phase: TEST\n\
  }\n\
  transform_param {\n\
    scale: 0.00390625\n\
  }\n\
  data_param {\n\
    source: \"examples/mnist/mnist_test_lmdb\"\n\
    batch_size: 100\n\
    backend: LMDB\n\
  }\n\
}\n"

def generate_conv_end_text(num_conv_layers):
    if (num_conv_layers == 0):
        text1 = "layer {\n\
  name: \"ip1\"\n\
  type: \"InnerProduct\"\n\
  bottom: \"data\"\n"
    else:
        text1 = "layer {\n\
  name: \"ip1\"\n\
  type: \"InnerProduct\"\n\
  bottom: \"pool" + str(num_conv_layers-1) + "\"\n"
      
        
    common_text_end = text1 + "  top: \"ip1\"\n\
  param {\n\
    lr_mult: 1\n\
  }\n\
  param {\n\
    lr_mult: 2\n\
  }\n\
  inner_product_param {\n\
    num_output: 500\n\
    weight_filler {\n\
      type: \"xavier\"\n\
    }\n\
    bias_filler {\n\
      type: \"constant\"\n\
    }\n\
  }\n\
}\n\
layer {\n\
  name: \"relu1\"\n\
  type: \"ReLU\"\n\
  bottom: \"ip1\"\n\
  top: \"ip1\"\n\
}\n\
layer {\n\
  name: \"ip2\"\n\
  type: \"InnerProduct\"\n\
  bottom: \"ip1\"\n\
  top: \"ip2\"\n\
  param {\n\
    lr_mult: 1\n\
  }\n\
  param {\n\
    lr_mult: 2\n\
    }\n\
  inner_product_param {\n\
    num_output: 10\n\
    weight_filler {\n\
      type: \"xavier\"\n\
    }\n\
    bias_filler {\n\
      type: \"constant\"\n\
    }\n\
  }\n\
}\n\
layer {\n\
  name: \"accuracy\"\n\
  type: \"Accuracy\"\n\
  bottom: \"ip2\"\n\
  bottom: \"label\"\n\
  top: \"accuracy\"\n\
  include {\n\
    phase: TEST\n\
  }\n\
}\n\
layer {\n\
  name: \"loss\"\n\
  type: \"SoftmaxWithLoss\"\n\
  bottom: \"ip2\"\n\
  bottom: \"label\"\n\
  top: \"loss\"\n\
 }\n"
    return common_text_end

def generate_conv_text(layer_id):
    if (layer_id == 0):
        text1 = "layer {\n\
  name: \"conv" + str(layer_id) + "\"\n\
  type: \"Convolution\"\n\
  bottom: \"data\"\n"
    else:   
        text1 = "layer {\n\
  name: \"conv" + str(layer_id) + "\"\n\
  type: \"Convolution\"\n\
  bottom: \"pool"  + str(layer_id-1) + "\"\n"

    text = text1 + "  top: \"conv"  + str(layer_id) + "\"\n\
  param {\n\
    lr_mult: 1\n\
  }\n\
  param {\n\
    lr_mult: 2\n\
  }\n\
  convolution_param {\n\
    num_output: 50\n\
    kernel_size: 5\n\
    stride: 1\n\
    weight_filler {\n\
      type: \"xavier\"\n\
    }\n\
    bias_filler {\n\
      type: \"constant\"\n\
    }\n\
  }\n\
}\n"
    return text


def generate_pool_text(layer_id):
    text = "layer {\n\
  name: \"pool" + str(layer_id) + "\"\n\
  type: \"Pooling\"\n\
  bottom: \"conv" + str(layer_id) + "\"\n\
  top: \"pool" + str(layer_id) + "\"\n\
  pooling_param {\n\
    pool: MAX\n\
    kernel_size: 2\n\
    stride: 2\n\
  }\n\
}\n"
    return text

num_conv = xrange(10) #no. of convolution layers

for i in xrange(num_trials):
    filename = 'lenet_train_test' + str(i) + '.prototxt'
    fid = open(filename, 'w')
    fid.write(common_text_start)
    for j in xrange(i):
        conv_text = generate_conv_text(j)
        fid.write(conv_text)
        pool_text = generate_pool_text(j)
        fid.write(pool_text)
    common_text_end = generate_conv_end_text(i)
    fid.write(common_text_end)
    fid.close()
    

    
