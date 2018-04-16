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
    net.mnist_image = L.Input(input_param=dict(shape=dict(dim=[1,1,1,784])), ntop=1)
    net.label = L.Input(input_param=dict(shape=dict(dim=[1,1,1,1])), ntop=1)
  
    net.hidden_layer = L.InnerProduct(net.mnist_image,
                                      bottom_data_type = data_type[0],
                                      compute_data_type = data_type[0],
                                      top_data_type = data_type[0],
                                      inner_product_param = dict(
                                      num_output = 30,
                                      weight_filler = dict(type='xavier'),
                                      bias_filler = dict(type='constant', value=0.0)))
    net.output_layer = L.InnerProduct(net.hidden_layer,
                                      bottom_data_type = data_type[0],
                                      compute_data_type = data_type[0],
                                      top_data_type = data_type[0], 
                                      inner_product_param = dict(
                                      num_output = 10,
                                      weight_filler = dict(type='xavier'),
                                      bias_filler = dict(type='constant', value=0.0)))
                                        
    net.loss = L.SoftmaxWithLoss(net.output_layer, net.label,include=dict(phase=0))
    net.pred = L.Softmax(net.output_layer, include=dict(phase=1))

    protonet = net.to_proto()
    protonet.name = 'net'
    with open(protonet.name + '_' + data_type[1] + '.prototxt', 'w') as f:
        print(protonet, file=f)
