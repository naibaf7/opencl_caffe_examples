import sys
sys.path.append('..')
from caffe_examples_setup import *

# We create the network for float, half, int16 and int8
data_types = [caffe.data_type.CAFFE_HALF, caffe.data_type.CAFFE_FLOAT,
              caffe.data_type.CAFFE_INT8_QUANTIZED, caffe.data_type.CAFFE_INT16_QUANTIZED]
data_types_names = ['half', 'float', 'int8', 'int16']

for data_type in zip(data_types, data_types_names):
    net = caffe.NetSpec()
    net.celsius = L.Input(input_param=dict(shape=dict(dim=[1,1,1,1])), ntop=1)
    net.farenheit = L.Input(input_param=dict(shape=dict(dim=[1,1,1,1])), ntop=1, include=dict(phase=0))
  
    net.neuron = L.InnerProduct(net.celsius,
                                bottom_data_type = data_type[0],
                                compute_data_type = data_type[0],
                                top_data_type = data_type[0],
                                inner_product_param = dict(num_output = 1,
                                          weight_filler = dict(type='constant'),
                                          bias_filler = dict(type='constant')))
    net.output = L.Quantizer(net.neuron,
                             bottom_data_type = data_type[0],
                             compute_data_type = data_type[0],
                             top_data_type = caffe.data_type.CAFFE_FLOAT)
    net.euclidean = L.EuclideanLoss(net.output, net.farenheit, include=dict(phase=0))

  
    protonet = net.to_proto()
    protonet.name = 'net'
    with open(protonet.name + '_' + data_type[1] + '.prototxt', 'w') as f:
        print(protonet, file=f)
