import itertools #for generating permutations

path = './train_test/var_everything_tiny/'

# Fucntions to generate layer specific text
common_text_start = "name: \"CIFAR\"\n\
layer {\n\
  name: \"cifar\"\n\
  type: \"Data\"\n\
  top: \"data\"\n\
  top: \"label\"\n\
  include {\n\
    phase: TRAIN\n\
  }\n\
  transform_param {\n\
    mean_file: \"examples/cifar10/mean.binaryproto\"\n\
  }\n\
  data_param {\n\
    source: \"examples/cifar10/cifar10_train_lmdb\"\n\
    batch_size: 111\n\
    backend: LMDB\n\
  }\n\
}\n\
layer {\n\
  name: \"cifar\"\n\
  type: \"Data\"\n\
  top: \"data\"\n\
  top: \"label\"\n\
  include {\n\
    phase: TEST\n\
  }\n\
  transform_param {\n\
    mean_file: \"examples/cifar10/mean.binaryproto\"\n\
  }\n\
  data_param {\n\
    source: \"examples/cifar10/cifar10_test_lmdb\"\n\
    batch_size: 1000\n\
    backend: LMDB\n\
  }\n\
}\n"

def generate_conv_text(layer_id, kernel_size, num_features):
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
    num_output: " + str(num_features) + "\n\
    pad: 2\n\
    kernel_size: " + str(kernel_size) + "\n\
    stride: 1\n\
    weight_filler {\n\
      type: \"gaussian\"\n\
      std: 0.01\n\
    }\n\
    bias_filler {\n\
      type: \"constant\"\n\
    }\n\
  }\n\
}\n"
    return text

def generate_sigmoid_text(layer_id):
    text = "layer {\n\
  name: \"Sigmoid" + str(layer_id) + "\"\n\
  type: \"Sigmoid\"\n\
  bottom: \"conv" + str(layer_id) + "\"\n\
  top: \"Sigmoid" + str(layer_id) + "\"\n\
}\n"
    return text

def generate_pool_text(layer_id):
    text = "layer {\n\
  name: \"pool" + str(layer_id) + "\"\n\
  type: \"Pooling\"\n\
  bottom: \"Sigmoid" + str(layer_id) + "\"\n\
  top: \"pool" + str(layer_id) + "\"\n\
  pooling_param {\n\
    pool: AVE\n\
    kernel_size: 3\n\
    stride: 2\n\
  }\n\
}\n"
    return text

def generate_relu_text(layer_id):
    text = "layer {\n\
  name: \"relu" + str(layer_id) + "\"\n\
  type: \"ReLU\"\n\
  bottom: \"ip" + str(layer_id) + "\"\n\
  top: \"ip" + str(layer_id) + "\"\n\
}\n"
    return text

def generate_fc_text(layer_id, num_output, num_conv):
    if (layer_id == 1):
        if (num_conv  == 0):
            bottom = 'data'
        else:
            bottom = 'pool' + str(num_conv-1)
    else:
        bottom = 'ip' + str(layer_id-1)

    text = "layer {\n\
  name: \"ip" + str(layer_id) + "\"\n\
  type: \"InnerProduct\"\n\
  bottom: \"" + bottom + "\"\n\
  top: \"ip" + str(layer_id) + "\"\n\
  param {\n\
    lr_mult: 1\n\
  }\n\
  param {\n\
    lr_mult: 2\n\
    }\n\
  inner_product_param {\n\
    num_output: " + str(num_output) + "\n\
    weight_filler {\n\
      type: \"gaussian\"\n\
      std: 0.01\n\
    }\n\
    bias_filler {\n\
      type: \"constant\"\n\
    }\n\
  }\n\
}\n"
    return text

def generate_fc_end_text (num_fc_layers, num_conv_layers):
    if (num_fc_layers == 0):
        bottom = "conv" + str(num_conv_layers-1) #conv layers start from 0
    else:
        bottom = "ip" + str(num_fc_layers)
    end_text = "layer {\n\
  name: \"accuracy\"\n\
  type: \"Accuracy\"\n\
  bottom: \"" + bottom + "\"\n\
  bottom: \"label\"\n\
  top: \"accuracy\"\n\
  include {\n\
    phase: TEST\n\
  }\n\
}\n\
layer {\n\
  name: \"loss\"\n\
  type: \"SoftmaxWithLoss\"\n\
  bottom: \"" + bottom + "\"\n\
  bottom: \"label\"\n\
  top: \"loss\"\n\
 }\n"
    return end_text

# Code to generate prototxt files
num_classes = 10

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
                    filename = path + 'cifar_train_test_' + str(index) + '.prototxt'
                    index = index + 1
                    fid = open(filename, 'w')

                    #file start text
                    fid.write(common_text_start)

                    #conv layers text (excluding input layer)
                    for j in xrange(num_conv_layers):
                        conv_text = generate_conv_text(j, kernel_tuple[j], feature_tuple[j])
                        fid.write(conv_text)
                        sigmoid_text = generate_sigmoid_text(j)
                        fid.write(sigmoid_text)
                        pool_text = generate_pool_text(j)
                        fid.write(pool_text)

                    #fc layers text (excluding output layer)
                    for k in xrange(num_fcl):
                        fc_text = generate_fc_text(k+1, fcn_tuple[k], num_conv_layers)
                        fid.write(fc_text)
                        relu_text = generate_relu_text(k+1)
                        fid.write(relu_text)

                    # generate output layer text
                    fc_text = generate_fc_text(num_fcl+1, num_classes, num_conv_layers)
                    fid.write(fc_text)
                    #end text
                    common_text_end = generate_fc_end_text(num_fcl+1, num_conv_layers)
                    fid.write(common_text_end)
                    fid.close()

print (index)



