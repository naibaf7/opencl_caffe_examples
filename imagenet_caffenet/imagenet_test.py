import sys
sys.path.append('..')
from caffe_examples_setup import *

# Choose the precision (half, float, int16 or int8)
precision = 'int8'
batch_size = 256

avg1 = 0.0
avg5 = 0.0

# Test how accurate the network has learned it's task
testnet = caffe.Net(str('net_' + precision + '_' + str(batch_size)  + '.prototxt'), caffe.TEST, weights='net_trained.caffemodel')
testnet.save('net_trained_'+precision+'.caffemodel')
for k in range(0, 100):
    testnet.forward()
    print(testnet.blobs['accuracy1'].data[:])
    print(testnet.blobs['accuracy5'].data[:])
    avg1 += testnet.blobs['accuracy1'].data[:]
    avg5 += testnet.blobs['accuracy5'].data[:]
    
print('Average:')
print(avg1/100.0)
print(avg5/100.0)
