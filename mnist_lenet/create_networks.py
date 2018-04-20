import sys
sys.path.append('..')
from caffe_examples_setup import *

# Create a simple network with just one hidden layer and a flat 784 size input vector

# We create the network for float, half, int16 and int8
data_types = [caffe.data_type.CAFFE_HALF, caffe.data_type.CAFFE_FLOAT,
              caffe.data_type.CAFFE_INT8_QUANTIZED, caffe.data_type.CAFFE_INT16_QUANTIZED]
data_types_names = ['half', 'float', 'int8', 'int16']

for data_type in zip(data_types, data_types_names):
    net = caffe.NetSpec()
    net.mnist_image = L.Input(input_param=dict(shape=dict(dim=[1,1,28,28])), ntop=1)
    net.label = L.Input(input_param=dict(shape=dict(dim=[1,1,1,1])), ntop=1)
    
    net.conv1 = L.Convolution(net.mnist_image,
                              bottom_data_type = data_type[0],
                              compute_data_type = data_type[0],
                              top_data_type = data_type[0],
                              param=[dict(lr_mult=1), dict(lr_mult=2)],
                              convolution_param=dict(num_output=20,kernel_size=5,stride=1,
                                                     weight_filler=dict(type='xavier'),
                                                     bias_filler=dict(type='constant')))
    net.pool1 = L.Pooling(net.conv1,
                          bottom_data_type = data_type[0],
                          compute_data_type = data_type[0],
                          top_data_type = data_type[0],
                          pooling_param=dict(pool=P.Pooling.MAX,kernel_size=2,stride=2))
    net.conv2 = L.Convolution(net.pool1,
                              bottom_data_type = data_type[0],
                              compute_data_type = data_type[0],
                              top_data_type = data_type[0],
                              param=[dict(lr_mult=1), dict(lr_mult=2)],
                              convolution_param=dict(num_output=50,kernel_size=5,stride=1,
                                                     weight_filler=dict(type='xavier'),
                                                     bias_filler=dict(type='constant')))
    net.pool2 = L.Pooling(net.conv2,
                          bottom_data_type = data_type[0],
                          compute_data_type = data_type[0],
                          top_data_type = data_type[0],
                          pooling_param=dict(pool=P.Pooling.MAX,kernel_size=2,stride=2))
    
    net.ip1 = L.InnerProduct(net.pool2,
                             bottom_data_type = data_type[0],
                             compute_data_type = data_type[0],
                             top_data_type = data_type[0],
                             param=[dict(lr_mult=1),dict(lr_mult=2)],
                             inner_product_param = dict(num_output = 500,
                                                        weight_filler = dict(type='xavier'),
                                                        bias_filler = dict(type='constant')))
    
    # Keep ReLU and second inner product at full precision
    net.relu1 = L.ReLU(net.ip1, in_place=False)
    net.ip2 = L.InnerProduct(net.relu1, param=[dict(lr_mult=1), dict(lr_mult=2)],
                             inner_product_param = dict(num_output = 10,
                                                        weight_filler = dict(type='xavier'),
                                                        bias_filler = dict(type='constant')))

    net.loss = L.SoftmaxWithLoss(net.ip2, net.label,include=dict(phase=0))
    net.pred = L.Softmax(net.ip2, include=dict(phase=1))

    protonet = net.to_proto()
    protonet.name = 'net'
    with open(protonet.name + '_' + data_type[1] + '.prototxt', 'w') as f:
       print(protonet, file=f)
