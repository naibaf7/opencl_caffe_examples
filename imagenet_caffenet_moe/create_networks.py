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
    
        # The MOE gating network
        gating_net = caffe.NetSpec()
    
        gating_net.ga_input = L.Input(input_param=dict(shape=dict(dim=[bss,48,27,27])), ntop=1,
                                      bottom_data_type = data_type[0],
                                      compute_data_type = data_type[0],
                                      top_data_type = data_type[0])
                                      
                                      
        gating_net.ga_conv2 = L.Convolution(gating_net.ga_input,
                                            bottom_data_type = data_type[0],
                                            compute_data_type = data_type[0],
                                            top_data_type = data_type[0],
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            convolution_param=dict(num_output=64, pad=2, kernel_size=5, group=2,
                                                                   weight_filler=dict(type='gaussian', std=0.01),
                                                                   bias_filler=dict(type='constant', value=0.5)))
        
        gating_net.ga_relu2 = L.ReLU(gating_net.ga_conv2, in_place=False,
                                     bottom_data_type = data_type[0],
                                     compute_data_type = data_type[0],
                                     top_data_type = data_type[0])
        
        gating_net.ga_pool2 = L.Pooling(gating_net.ga_relu2,
                                        bottom_data_type = data_type[0],
                                        compute_data_type = data_type[0],
                                        top_data_type = data_type[0],
                                        pooling_param=dict(pool=P.Pooling.MAX, kernel_size=3, stride=2))
        
        gating_net.ga_norm2 = L.LRN(gating_net.ga_pool2,
                                    engine = P.LRN.CAFFE,
                                    bottom_data_type = caffe.data_type.CAFFE_FLOAT,
                                    compute_data_type = caffe.data_type.CAFFE_FLOAT,
                                    top_data_type = caffe.data_type.CAFFE_FLOAT,
                                    lrn_param=dict(local_size=5, alpha=0.0001, beta=0.75))
    
        gating_net.ga_fc1 = L.InnerProduct(gating_net.ga_norm2,
                                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                           inner_product_param = dict(num_output=128,
                                           weight_filler = dict(type='gaussian', std=0.01),
                                           bias_filler = dict(type='constant', value=0.5)),
                                           bottom_data_type = data_type[0],
                                           compute_data_type = data_type[0],
                                           top_data_type = data_type[0])
    
                
        gating_net.ga_relu3 = L.ReLU(gating_net.ga_fc1, in_place=False,
                                     bottom_data_type = data_type[0],
                                     compute_data_type = data_type[0],
                                     top_data_type = data_type[0])
    
    
        gating_net.ga_fc2 = L.InnerProduct(gating_net.ga_relu3,
                                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                                           inner_product_param = dict(num_output=16,
                                           weight_filler = dict(type='gaussian', std=0.01),
                                           bias_filler = dict(type='constant', value=0.0)),
                                           bottom_data_type = data_type[0],
                                           compute_data_type = data_type[0],
                                           top_data_type = data_type[0])
                                                                                  
        gating_net.ga_fc3 = L.InnerProduct(gating_net.ga_relu3,
                                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                                           inner_product_param = dict(num_output=16,
                                           weight_filler = dict(type='gaussian', std=0.01),
                                           bias_filler = dict(type='constant', value=0.0)),
                                           bottom_data_type = data_type[0],
                                           compute_data_type = data_type[0],
                                           top_data_type = data_type[0])
                                       
        gating_net.ga_relu4 = L.ReLU(gating_net.ga_fc3, in_place=False)
    
        gating_net.ga_noise = L.Noise(gating_net.ga_relu4, noise_param=dict(mu=0.0, sigma=1.0))
    
        gating_net.ga_eltwise1 = L.Eltwise(gating_net.ga_fc2, gating_net.ga_noise)
        
        gating_net.ga_dummy = L.DummyData(data_filler=dict(type='gaussian', std=10.0),
                                          dummy_data_param = dict(shape=dict(dim=[bss,16,1,1])))
        
        gating_net.ga_dummy_conv = L.Convolution(gating_net.ga_dummy,
                                                 bottom_data_type = data_type[0],
                                                 compute_data_type = data_type[0],
                                                 top_data_type = data_type[0],
                                                 param=[dict(lr_mult=0.00001, decay_mult=1.0), dict(lr_mult=0, decay_mult=0)],
                                                 convolution_param=dict(num_output=16, pad=0, kernel_size=1, group=16,
                                                                      weight_filler=dict(type='constant', value=1.0),
                                                                      bias_filler=dict(type='constant', value=0.0)))
          
        gating_net.ga_dummy_reshaped = L.Reshape(gating_net.ga_dummy_conv, reshape_param = dict(shape=dict(dim=[bss,16])))
        
        gating_net.ga_eltwise2 = L.Eltwise(gating_net.ga_eltwise1, gating_net.ga_dummy_reshaped)
            
        gating_net.ga_softmax = L.Softmax(gating_net.ga_eltwise2)
    
    
        expert_nets = [caffe.NetSpec() for i in range(0, 16)]
        # The expert networks
        exidx = 0
        for expert_net in expert_nets:
    
            setattr(expert_net, 'ex_'+str(exidx)+'_input', L.Input(input_param=dict(shape=dict(dim=[1,48,27,27])), ntop=1,
                                                                   bottom_data_type = data_type[0],
                                                                   compute_data_type = data_type[0],
                                                                   top_data_type = data_type[0]))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_conv2', L.Convolution(getattr(expert_net, 'ex_'+str(exidx)+'_input'),
                                                                         bottom_data_type = data_type[0],
                                                                         compute_data_type = data_type[0],
                                                                         top_data_type = data_type[0],
                                                                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                                                         convolution_param=dict(num_output=64, pad=2, kernel_size=5, group=2,
                                                                                                weight_filler=dict(type='gaussian', std=0.01),
                                                                                                bias_filler=dict(type='constant', value=0.5))))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_relu2', L.ReLU(getattr(expert_net, 'ex_'+str(exidx)+'_conv2'), in_place=False,
                                                                  bottom_data_type = data_type[0],
                                                                  compute_data_type = data_type[0],
                                                                  top_data_type = data_type[0]))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_pool2', L.Pooling(getattr(expert_net, 'ex_'+str(exidx)+'_relu2'),
                                                                     bottom_data_type = data_type[0],
                                                                     compute_data_type = data_type[0],
                                                                     top_data_type = data_type[0],
                                                                     pooling_param=dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_norm2', L.LRN(getattr(expert_net, 'ex_'+str(exidx)+'_pool2'),
                                                                 engine = P.LRN.CAFFE,
                                                                 bottom_data_type = caffe.data_type.CAFFE_FLOAT,
                                                                 compute_data_type = caffe.data_type.CAFFE_FLOAT,
                                                                 top_data_type = caffe.data_type.CAFFE_FLOAT,
                                                                 lrn_param=dict(local_size=5, alpha=0.0001, beta=0.75)))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_conv3', L.Convolution(getattr(expert_net, 'ex_'+str(exidx)+'_norm2'),
                                                                         bottom_data_type = data_type[0],
                                                                         compute_data_type = data_type[0],
                                                                         top_data_type = data_type[0],
                                                                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                                                         convolution_param=dict(num_output=96, pad=1, kernel_size=3,
                                                                                                weight_filler=dict(type='gaussian', std=0.01),
                                                                                                bias_filler=dict(type='constant', value=0.0))))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_relu3', L.ReLU(getattr(expert_net, 'ex_'+str(exidx)+'_conv3'), in_place=False,
                                                                  bottom_data_type = data_type[0],
                                                                  compute_data_type = data_type[0],
                                                                  top_data_type = data_type[0]))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_conv4', L.Convolution(getattr(expert_net, 'ex_'+str(exidx)+'_relu3'),
                                                                         bottom_data_type = data_type[0],
                                                                         compute_data_type = data_type[0],
                                                                         top_data_type = data_type[0],
                                                                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                                                         convolution_param=dict(num_output=96, pad=1, kernel_size=3, group=2,
                                                                                                weight_filler=dict(type='gaussian', std=0.01),
                                                                                                bias_filler=dict(type='constant', value=0.5))))
          
            setattr(expert_net, 'ex_'+str(exidx)+'_relu4', L.ReLU(getattr(expert_net, 'ex_'+str(exidx)+'_conv4'), in_place=False,
                                                                  bottom_data_type = data_type[0],
                                                                  compute_data_type = data_type[0],
                                                                  top_data_type = data_type[0]))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_conv5', L.Convolution(getattr(expert_net, 'ex_'+str(exidx)+'_relu4'),
                                                                         bottom_data_type = data_type[0],
                                                                         compute_data_type = data_type[0],
                                                                         top_data_type = data_type[0],
                                                                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                                                         convolution_param=dict(num_output=64, pad=1, kernel_size=3, group=2,
                                                                                                weight_filler=dict(type='gaussian', std=0.01),
                                                                                                bias_filler=dict(type='constant', value=0.5))))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_relu5', L.ReLU(getattr(expert_net, 'ex_'+str(exidx)+'_conv5'), in_place=False,
                                                                  bottom_data_type = data_type[0],
                                                                  compute_data_type = data_type[0],
                                                                  top_data_type = data_type[0]))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_pool5', L.Pooling(getattr(expert_net, 'ex_'+str(exidx)+'_relu5'),
                                                                     bottom_data_type = data_type[0],
                                                                     compute_data_type = data_type[0],
                                                                     top_data_type = data_type[0],
                                                                     pooling_param=dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)))

            setattr(expert_net, 'ex_'+str(exidx)+'_fc6', L.InnerProduct(getattr(expert_net, 'ex_'+str(exidx)+'_pool5'),
                                                                        bottom_data_type = data_type[0],
                                                                        compute_data_type = data_type[0],
                                                                        top_data_type = data_type[0],
                                                                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                                                        inner_product_param = dict(num_output=1024,
                                                                                                   weight_filler = dict(type='gaussian', std=0.01),
                                                                                                   bias_filler = dict(type='constant', value=0.5))))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_relu6', L.ReLU(getattr(expert_net, 'ex_'+str(exidx)+'_fc6'), in_place=False,
                                                                  bottom_data_type = data_type[0],
                                                                  compute_data_type = data_type[0],
                                                                  top_data_type = data_type[0]))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_drop6', L.Dropout(getattr(expert_net, 'ex_'+str(exidx)+'_relu6'), in_place=True,
                                                                     bottom_data_type = data_type[0],
                                                                     compute_data_type = data_type[0],
                                                                     top_data_type = data_type[0],
                                                                     dropout_param = dict(dropout_ratio=0.5)))
            
            setattr(expert_net, 'ex_'+str(exidx)+'_fc7', L.InnerProduct(getattr(expert_net, 'ex_'+str(exidx)+'_relu6'),
                                                                        bottom_data_type = data_type[0],
                                                                        compute_data_type = data_type[0],
                                                                        top_data_type = data_type[0],
                                                                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                                                        inner_product_param = dict(num_output=2048,
                                                                                                   weight_filler = dict(type='gaussian', std=0.01),
                                                                                                   bias_filler = dict(type='constant', value=0.5))))

                                                                        
            setattr(expert_net, 'ex_'+str(exidx)+'_qu', L.Quantizer(getattr(expert_net, 'ex_'+str(exidx)+'_fc7'),
                                                                    bottom_data_type = data_type[0],
                                                                    compute_data_type = data_type[0],
                                                                    top_data_type = caffe.data_type.CAFFE_FLOAT))
            exidx += 1
    
    
        net = caffe.NetSpec()

        
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
                                  convolution_param=dict(num_output=48, kernel_size=11, stride=4,
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
                          engine = P.LRN.CAFFE,
                          bottom_data_type = caffe.data_type.CAFFE_FLOAT,
                          compute_data_type = caffe.data_type.CAFFE_FLOAT,
                          top_data_type = caffe.data_type.CAFFE_FLOAT,
                          lrn_param=dict(local_size=5, alpha=0.0001, beta=0.75))
                                
        
        gating_net_proto = gating_net.to_proto()
        gating_net_proto.reduced_memory_inference = True
        expert_net_protos = [expert_net.to_proto() for expert_net in expert_nets]
        for expert_net in expert_net_protos:
            expert_net.reduced_memory_inference = True
        
        net.moe, net.observed_count, net.expected_count = L.MOE(net.norm1, ntop=3,
                                                                bottom_data_type = data_type[0],
                                                                compute_data_type = data_type[0],
                                                                top_data_type = caffe.data_type.CAFFE_FLOAT,
                                                                moe_param=dict(select_experts=4, gating_net=gating_net_proto, expert_net=expert_net_protos,
                                                                               full_forward=True))
        
        net.reg = L.EuclideanLoss(net.observed_count, net.expected_count, include=dict(phase=0), loss_weight=15.0)
        net.silence = L.Silence(net.observed_count, net.expected_count, include=dict(phase=1), ntop=0)
        
        net.relu7 = L.ReLU(net.moe, in_place=False,
                           bottom_data_type = data_type[0],
                           compute_data_type = data_type[0],
                           top_data_type = data_type[0])
        
        net.drop7 = L.Dropout(net.relu7, in_place=True,
                              bottom_data_type = data_type[0],
                              compute_data_type = data_type[0],
                              top_data_type = data_type[0],
                              dropout_param = dict(dropout_ratio=0.5))
        
        net.fc8 = L.InnerProduct(net.relu7,
                                 bottom_data_type = data_type[0],
                                 compute_data_type = data_type[0],
                                 top_data_type = data_type[0],
                                 param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                 inner_product_param = dict(num_output = 1000,
                                                            weight_filler = dict(type='gaussian', std=0.01),
                                                            bias_filler = dict(type='constant', value=0.0)))
        
        net.loss = L.SoftmaxWithLoss(net.fc8, net.label, include=dict(phase=0))
        net.accuracy1 = L.Accuracy(net.fc8, net.label, include=dict(phase=1), accuracy_param=dict(top_k=1))
        net.accuracy5 = L.Accuracy(net.fc8, net.label, include=dict(phase=1), accuracy_param=dict(top_k=5))
        
        protonet = net.to_proto()
        protonet.name = 'net'
        protonet.reduced_memory_inference = True
        with open(protonet.name + '_' + data_type[1] + '_' + str(bss) + '.prototxt', 'w') as f:
            print(protonet, file=f)
