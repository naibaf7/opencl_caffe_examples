import sys
sys.path.append('..')
from caffe_examples_setup import *

# Create a simple network with just one hidden layer and a flat 784 size input vector

# We create the network for float, half, int16 and int8
data_types = [caffe.data_type.CAFFE_HALF, caffe.data_type.CAFFE_FLOAT,
              caffe.data_type.CAFFE_INT8_QUANTIZED, caffe.data_type.CAFFE_INT16_QUANTIZED]
data_types_names = ['half', 'float', 'int8', 'int16']

batch_size = 128

for data_type in zip(data_types, data_types_names):

    # The MOE gating network
    gating_net = caffe.NetSpec()
    
    gating_net.ga_input = L.Input(input_param=dict(shape=dict(dim=[1,10,12,12])), ntop=1,
                                  bottom_data_type = data_type[0],
                                  compute_data_type = data_type[0],
                                  top_data_type = data_type[0])
        
    gating_net.ga_ip1 = L.InnerProduct(gating_net.ga_input,
                                       bottom_data_type = data_type[0],
                                       compute_data_type = data_type[0],
                                       top_data_type = data_type[0],
                                       param=[dict(lr_mult=1),dict(lr_mult=2)],
                                       inner_product_param = dict(num_output=25,
                                                                  weight_filler = dict(type='xavier'),
                                                                  bias_filler = dict(type='constant')))
    
    # Keep gating stage at full precision
    gating_net.ga_relu1 = L.ReLU(gating_net.ga_ip1, in_place=False)
    
    gating_net.ga_ip2 = L.InnerProduct(gating_net.ga_relu1, param=[dict(lr_mult=1), dict(lr_mult=2)],
                                       inner_product_param = dict(num_output=9,
                                       weight_filler = dict(type='constant', value=0.1),
                                       bias_filler = dict(type='constant', value=0.0)))
                                       
    gating_net.ga_ip3 = L.InnerProduct(gating_net.ga_relu1, param=[dict(lr_mult=1), dict(lr_mult=0)],
                                       inner_product_param = dict(num_output=9,
                                       weight_filler = dict(type='constant', value=0.1),
                                       bias_filler = dict(type='constant', value=0.0)))
                                       
    gating_net.ga_relu2 = L.ReLU(gating_net.ga_ip3, in_place=False)
    
    gating_net.ga_noise = L.Noise(gating_net.ga_relu2, noise_param=dict(mu=0.0, sigma=0.1))
    
    gating_net.ga_eltwise = L.Eltwise(gating_net.ga_ip2, gating_net.ga_noise)
    
    gating_net.ga_softmax = L.Softmax(gating_net.ga_eltwise)
    
    expert_nets = [caffe.NetSpec() for i in range(0, 9)]
    # The expert networks
    exidx = 0
    for expert_net in expert_nets:
    
        setattr(expert_net, 'ex_'+str(exidx)+'_input', L.Input(input_param=dict(shape=dict(dim=[1,10,12,12])), ntop=1,
                                                               bottom_data_type = data_type[0],
                                                               compute_data_type = data_type[0],
                                                               top_data_type = data_type[0]))
        
        setattr(expert_net, 'ex_'+str(exidx)+'_conv2', L.Convolution(getattr(expert_net, 'ex_'+str(exidx)+'_input'),
                                                                     bottom_data_type = data_type[0],
                                                                     compute_data_type = data_type[0],
                                                                     top_data_type = data_type[0],
                                                                     param=[dict(lr_mult=1), dict(lr_mult=2)],
                                                                     convolution_param=dict(num_output=5,kernel_size=5,stride=1,
                                                                                            weight_filler=dict(type='xavier'),
                                                                                            bias_filler=dict(type='constant'))))
                                                                                            
        setattr(expert_net, 'ex_'+str(exidx)+'_pool2', L.Pooling(getattr(expert_net, 'ex_'+str(exidx)+'_conv2'),
                                                            bottom_data_type = data_type[0],
                                                            compute_data_type = data_type[0],
                                                            top_data_type = data_type[0],
                                                            pooling_param=dict(pool=P.Pooling.MAX,kernel_size=2,stride=2)))
                                  
        setattr(expert_net, 'ex_'+str(exidx)+'_ip2', L.InnerProduct(getattr(expert_net, 'ex_'+str(exidx)+'_pool2'),
                                                               bottom_data_type = data_type[0],
                                                               compute_data_type = data_type[0],
                                                               top_data_type = data_type[0],
                                                               param=[dict(lr_mult=1),dict(lr_mult=2)],
                                                               inner_product_param = dict(num_output = 25,
                                                                                          weight_filler = dict(type='xavier'),
                                                                                          bias_filler = dict(type='constant'))))
                                                                                          
        setattr(expert_net, 'ex_'+str(exidx)+'_qu', L.Quantizer(getattr(expert_net, 'ex_'+str(exidx)+'_ip2'),
                                                                        bottom_data_type = data_type[0],
                                                                        compute_data_type = data_type[0],
                                                                        top_data_type = caffe.data_type.CAFFE_FLOAT))
        
        exidx += 1


    net = caffe.NetSpec()
    net.mnist_image = L.Input(input_param=dict(shape=dict(dim=[batch_size,1,28,28])), ntop=1)
    net.label = L.Input(input_param=dict(shape=dict(dim=[batch_size,1,1,1])), ntop=1)
    
    net.conv1 = L.Convolution(net.mnist_image,
                              bottom_data_type = data_type[0],
                              compute_data_type = data_type[0],
                              top_data_type = data_type[0],
                              param=[dict(lr_mult=1), dict(lr_mult=2)],
                              convolution_param=dict(num_output=10,kernel_size=5,stride=1,
                                                     weight_filler=dict(type='xavier'),
                                                     bias_filler=dict(type='constant')))
    net.pool1 = L.Pooling(net.conv1,
                          bottom_data_type = data_type[0],
                          compute_data_type = data_type[0],
                          top_data_type = data_type[0],
                          pooling_param=dict(pool=P.Pooling.MAX,kernel_size=2,stride=2))

    gating_net_proto = gating_net.to_proto()
    expert_net_protos = [expert_net.to_proto() for expert_net in expert_nets]
    
    net.moe, net.observed_count, net.expected_count = L.MOE(net.pool1, ntop=3,
                                                            bottom_data_type = data_type[0],
                                                            compute_data_type = data_type[0],
                                                            top_data_type = caffe.data_type.CAFFE_FLOAT,
                                                            moe_param=dict(select_experts=3, gating_net=gating_net_proto, expert_net=expert_net_protos))
                                                            
    net.reg = L.EuclideanLoss(net.observed_count, net.expected_count, include=dict(phase=0), loss_weight=50.0)
    net.silence = L.Silence(net.observed_count, net.expected_count, include=dict(phase=1), ntop=0)

    # Keep ReLU and second inner product at full precision
    net.relu1 = L.ReLU(net.moe, in_place=False)
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
