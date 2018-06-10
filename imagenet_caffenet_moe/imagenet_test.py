import sys
sys.path.append('..')
from caffe_examples_setup import *

# Choose the precision (half, float, int16 or int8)
precision = 'float'
batch_size = 256

# Test how accurate the network has learned it's task
testnet = caffe.Net(str('net_' + precision + '_' + str(batch_size)  + '.prototxt'), caffe.TEST, weights='net_trained.caffemodel')
for k in range(0, 1000):
    testnet.forward()
    print(testnet.blobs['accuracy'].data[:])
