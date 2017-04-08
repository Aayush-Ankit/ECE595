num_trials = 10
#path = './train_test/var_ker_size/'
#path = './train_test/var_fc_size/'
path = './train_test/var_num_feature/'

# Fucntions to generate layer specific text
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
    kernel_size: " + str(kernel_size) + "\n\
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
      type: \"xavier\"\n\
    }\n\
    bias_filler {\n\
      type: \"constant\"\n\
    }\n\
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


## Code to generate prototxt files
num_conv = xrange(10) #no. of convolution layers

#variable number of convolutional layers with kernel size 5
##for i in xrange(num_trials):
##    filename = 'lenet_train_test' + str(i) + '.prototxt'
##    fid = open(filename, 'w')
##    fid.write(common_text_start)
##    for j in xrange(i):
##        conv_text = generate_conv_text(j, 5)
##        fid.write(conv_text)
##        pool_text = generate_pool_text(j)
##        fid.write(pool_text)
##    common_text_end = generate_conv_end_text(i)
##    fid.write(common_text_end)
##    fid.close()
    
#variable kernel sizes for fixed 2 convolutional layers
num_conv_layers = 2
num_classes = 10
ker_size = 5
##kernel_size_list = [3, 5]
##for k_size in kernel_size_list:
##    filename = path + 'lenet_train_test_kersize' + str(k_size) + '.prototxt'
##    fid = open(filename, 'w')
##    fid.write(common_text_start)
##    for j in xrange(num_layers):
##        conv_text = generate_conv_text(j, k_size)
##        fid.write(conv_text)
##        pool_text = generate_pool_text(j)
##        fid.write(pool_text)
##    common_text_end = generate_conv_end_text(num_layers)
##    fid.write(common_text_end)
##    fid.close()

num_fc_layer = 2
num_fc_neurons = 100
##num_fc_layer = [1, 2, 3] #num FC layers
##num_fc_neurons = [100, 300, 500] #size of FC layer
##for num_fcl in num_fc_layer:
##    for num_fcn in num_fc_neurons:
##        filename = path + 'lenet_train_test_fcl' + str(num_fcl) + '_fcn' + str(num_fcn) + '.prototxt'
##        fid = open(filename, 'w')
##        fid.write(common_text_start)
##        for j in xrange(num_conv_layers):
##            conv_text = generate_conv_text(j, ker_size)
##            fid.write(conv_text)
##            pool_text = generate_pool_text(j)
##            fid.write(pool_text)
##        for k in xrange(num_fcl):
##            if (k+1 == num_fcl):
##                fc_text = generate_fc_text(k+1, num_classes, num_conv_layers)
##                fid.write(fc_text)
##            else:
##                fc_text = generate_fc_text(k+1, num_fcn, num_conv_layers)
##                fid.write(fc_text)
##            relu_text = generate_relu_text(k+1)
##            fid.write(relu_text)
##        end_text = generate_fc_end_text(num_fcl,num_conv_layers)
##        fid.write(end_text)
##        fid.close()


#variable number of convolutional features
feature_size_list = [10, 25, 50, 100, 150]
for f_size in feature_size_list:
    filename = path + 'lenet_train_test_featuresize' + str(f_size) + '.prototxt'
    fid = open(filename, 'w')
    fid.write(common_text_start)
    for j in xrange(num_conv_layers):
        conv_text = generate_conv_text(j, ker_size, f_size)
        fid.write(conv_text)
        pool_text = generate_pool_text(j)
        fid.write(pool_text)
    for k in xrange(num_fc_layer):
       if (k+1 == num_fc_layer):
           fc_text = generate_fc_text(k+1, num_classes, num_conv_layers)
           fid.write(fc_text)
       else:
           fc_text = generate_fc_text(k+1, num_fc_neurons, num_conv_layers)
           fid.write(fc_text)
       relu_text = generate_relu_text(k+1)
       fid.write(relu_text)
    
    common_text_end = generate_fc_end_text(num_fc_layer, num_conv_layers)
    fid.write(common_text_end)
    fid.close()
            
        
    
        
