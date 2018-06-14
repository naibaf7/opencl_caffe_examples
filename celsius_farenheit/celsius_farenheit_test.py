import sys
sys.path.append('..')
from caffe_examples_setup import *

# Choose the precision (half, float, int16 or int8)
precision = 'int8'

# Define the training and testing data
values_celsius = np.array([(float)(c) for c in range(-273,1000)])
# We know that farenheit = celsius * 1.8 + 32.0
values_farenheit = np.array([c*1.8+32.0 for c in values_celsius])

# Split data into training (90%) and testing (10%)
indices = np.random.permutation(values_celsius.shape[0])
training_idx, test_idx = indices[:(int)(90*values_celsius.shape[0]/100)], indices[(int)(90*values_celsius.shape[0]/100):]

values_celsius_train = values_celsius[training_idx]
values_farenheit_train = values_farenheit[training_idx]
values_celsius_test = values_celsius[test_idx]
values_farenheit_test = values_farenheit[test_idx]

# Test how accurate the network has learned it's task
error = []
testnet = caffe.Net(str('net_' + precision + '.prototxt'), caffe.TEST, weights='net_trained.caffemodel')
for c,f in zip(values_celsius_test,values_farenheit_test):
    testnet.blobs['celsius'].data[0] = c
    testnet.forward()
    predicted_f = testnet.blobs['output'].data[0,0]
    print('Cesius: '+str(c)+'°C, predicted: '+str(predicted_f)+' °F, actual: '+str(f)+' °F')
    error.append(abs(f-predicted_f))
print('Average error: '+str(np.array(error).mean())+' °F')
