import sys
sys.path.append('..')
from caffe_examples_setup import *

# Create a simple network with just one hidden layer and a flat 784 size input vector

# We create the network for float, half, int16 and int8
data_types = [caffe.data_type.CAFFE_HALF, caffe.data_type.CAFFE_FLOAT,
              caffe.data_type.CAFFE_INT8_QUANTIZED, caffe.data_type.CAFFE_INT16_QUANTIZED]
data_types_names = ['half', 'float', 'int8', 'int16']

for bs in range(0, 14):
    bss = int(math.pow(2, bs))
    for data_type in zip(data_types, data_types_names):
        net = caffe.NetSpec()
        # net.data = L.Input(input_param=dict(shape=dict(dim=[bss,3,227,227])), ntop=1)
        # net.label = L.Input(input_param=dict(shape=dict(dim=[bss,1,1,1])), ntop=1)
        
        train_data, train_label = L.Data(bottom_data_type = caffe.data_type.CAFFE_FLOAT,
                                         compute_data_type = caffe.data_type.CAFFE_FLOAT,
                                         top_data_type = caffe.data_type.CAFFE_FLOAT,
                                         data_param=dict(source=imagenet_data+'/ilsvrc12_train_lmdb',
                                                         batch_size=bss, backend=P.Data.LMDB),
                                         transform_param=dict(mirror=True, crop_size=227, mean_file=imagenet_data+'/imagenet_mean.binaryproto'),
                                         ntop=2, include=dict(phase=0))
        
        test_data, test_label = L.Data(bottom_data_type = caffe.data_type.CAFFE_FLOAT,
                                       compute_data_type = caffe.data_type.CAFFE_FLOAT,
                                       top_data_type = caffe.data_type.CAFFE_FLOAT,
                                       data_param=dict(source=imagenet_data+'/ilsvrc12_val_lmdb',
                                                       batch_size=bss, backend=P.Data.LMDB),
                                       transform_param=dict(mirror=False, crop_size=227, mean_file=imagenet_data+'/imagenet_mean.binaryproto'),
                                       ntop=2, include=dict(phase=1))
        
        net.data = [train_data, test_data]
        net.label = [train_label, test_label]
        
        net.conv1 = L.Convolution(net.data,
                                  bottom_data_type = data_type[0],
                                  compute_data_type = data_type[0],
                                  top_data_type = data_type[0],
                                  param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                  convolution_param=dict(num_output=96, kernel_size=11, stride=4,
                                                         weight_filler=dict(type='gaussian', std=0.01),
                                                         bias_filler=dict(type='constant', value=0.0)))
        
        net.relu1 = L.ReLU(net.conv1, in_place=False,
                           bottom_data_type = data_type[0],
                           compute_data_type = data_type[0],
                           top_data_type = data_type[0])
        
        net.pool1 = L.Pooling(net.relu1,
                              bottom_data_type = data_type[0],
                              compute_data_type = data_type[0],
                              top_data_type = data_type[0],
                              pooling_param=dict(pool=P.Pooling.MAX, kernel_size=3, stride=2))
      
        net.norm1 = L.LRN(net.pool1,
                          bottom_data_type = caffe.data_type.CAFFE_FLOAT, # if 'int' in data_type[1] else data_type[0],
                          compute_data_type = caffe.data_type.CAFFE_FLOAT, # if 'int' in data_type[1] else data_type[0],
                          top_data_type = caffe.data_type.CAFFE_FLOAT, # if 'int' in data_type[1] else data_type[0],
                          lrn_param=dict(local_size=5, alpha=0.0001, beta=0.75))
        
        net.conv2 = L.Convolution(net.norm1,
                                  bottom_data_type = data_type[0],
                                  compute_data_type = data_type[0],
                                  top_data_type = data_type[0],
                                  param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                  convolution_param=dict(num_output=256, pad=2, kernel_size=5, group=2,
                                                         weight_filler=dict(type='gaussian', std=0.01),
                                                         bias_filler=dict(type='constant', value=1.0)))
        
        net.relu2 = L.ReLU(net.conv2, in_place=False,
                           bottom_data_type = data_type[0],
                           compute_data_type = data_type[0],
                           top_data_type = data_type[0])
        
        net.pool2 = L.Pooling(net.relu2,
                              bottom_data_type = data_type[0],
                              compute_data_type = data_type[0],
                              top_data_type = data_type[0],
                              pooling_param=dict(pool=P.Pooling.MAX, kernel_size=3, stride=2))
        
        net.norm2 = L.LRN(net.pool2,
                          bottom_data_type = caffe.data_type.CAFFE_FLOAT, # if 'int' in data_type[1] else data_type[0],
                          compute_data_type = caffe.data_type.CAFFE_FLOAT, # if 'int' in data_type[1] else data_type[0],
                          top_data_type = caffe.data_type.CAFFE_FLOAT, # if 'int' in data_type[1] else data_type[0],
                          lrn_param=dict(local_size=5, alpha=0.0001, beta=0.75))
        
        net.conv3 = L.Convolution(net.norm2,
                                  bottom_data_type = data_type[0],
                                  compute_data_type = data_type[0],
                                  top_data_type = data_type[0],
                                  param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                  convolution_param=dict(num_output=384, pad=1, kernel_size=3,
                                                         weight_filler=dict(type='gaussian', std=0.01),
                                                         bias_filler=dict(type='constant', value=0.0)))
        
        net.relu3 = L.ReLU(net.conv3, in_place=False,
                           bottom_data_type = data_type[0],
                           compute_data_type = data_type[0],
                           top_data_type = data_type[0])
        
        net.conv4 = L.Convolution(net.relu3,
                                  bottom_data_type = data_type[0],
                                  compute_data_type = data_type[0],
                                  top_data_type = data_type[0],
                                  param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                  convolution_param=dict(num_output=384, pad=1, kernel_size=3, group=2,
                                                         weight_filler=dict(type='gaussian', std=0.01),
                                                         bias_filler=dict(type='constant', value=1.0)))
      
        net.relu4 = L.ReLU(net.conv4, in_place=False,
                           bottom_data_type = data_type[0],
                           compute_data_type = data_type[0],
                           top_data_type = data_type[0])
        
        net.conv5 = L.Convolution(net.relu4,
                                  bottom_data_type = data_type[0],
                                  compute_data_type = data_type[0],
                                  top_data_type = data_type[0],
                                  param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                  convolution_param=dict(num_output=256, pad=1, kernel_size=3, group=2,
                                                         weight_filler=dict(type='gaussian', std=0.01),
                                                         bias_filler=dict(type='constant', value=1.0)))
        
        net.relu5 = L.ReLU(net.conv5, in_place=False,
                           bottom_data_type = data_type[0],
                           compute_data_type = data_type[0],
                           top_data_type = data_type[0])
        
        net.pool5 = L.Pooling(net.relu5,
                              bottom_data_type = data_type[0],
                              compute_data_type = data_type[0],
                              top_data_type = data_type[0],
                              pooling_param=dict(pool=P.Pooling.MAX, kernel_size=3, stride=2))
      
        net.fc6 = L.InnerProduct(net.pool5,
                                 bottom_data_type = data_type[0],
                                 compute_data_type = data_type[0],
                                 top_data_type = data_type[0],
                                 param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                 inner_product_param = dict(
                                     num_output = 4096,
                                     weight_filler = dict(type='gaussian', std=0.005),
                                     bias_filler = dict(type='constant', value=1.0)))
        
        net.relu6 = L.ReLU(net.fc6, in_place=False,
                           bottom_data_type = data_type[0],
                           compute_data_type = data_type[0],
                           top_data_type = data_type[0])
        
        net.drop6 = L.Dropout(net.relu6,
                              in_place = False,
                              bottom_data_type = data_type[0],
                              compute_data_type = data_type[0],
                              top_data_type = data_type[0],
                              dropout_param = dict(dropout_ratio=0.5))
        
        net.fc7 = L.InnerProduct(net.drop6,
                                 bottom_data_type = data_type[0],
                                 compute_data_type = data_type[0],
                                 top_data_type = data_type[0],
                                 param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                 inner_product_param = dict(
                                     num_output = 4096,
                                     weight_filler = dict(type='gaussian', std=0.005),
                                     bias_filler = dict(type='constant', value=1.0)))
        
        net.relu7 = L.ReLU(net.fc7, in_place=False,
                           bottom_data_type = data_type[0],
                           compute_data_type = data_type[0],
                           top_data_type = data_type[0])
        
        net.drop7 = L.Dropout(net.relu7, in_place=False,
                              bottom_data_type = data_type[0],
                              compute_data_type = data_type[0],
                              top_data_type = data_type[0],
                              dropout_param = dict(dropout_ratio=0.5))
        
        net.fc8 = L.InnerProduct(net.drop7,
                                 bottom_data_type = data_type[0],
                                 compute_data_type = data_type[0],
                                 top_data_type = data_type[0],
                                 param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                 inner_product_param = dict(
                                     num_output = 1000,
                                     weight_filler = dict(type='gaussian', std=0.01),
                                     bias_filler = dict(type='constant', value=0.0)))
        
        net.loss = L.SoftmaxWithLoss(net.fc8, net.label, include=dict(phase=0))
        # net.pred = L.Softmax(net.fc8, net.label, include=dict(phase=1))
        net.accuracy = L.Accuracy(net.fc8, net.label, include=dict(phase=1), accuracy_param=dict(top_k=5))

        protonet = net.to_proto()
        protonet.name = 'net'
        with open(protonet.name + '_' + data_type[1] + '_' + str(bss) + '.prototxt', 'w') as f:
            print(protonet, file=f)
