import sys
sys.path.append('..')
from caffe_examples_setup import *

# Choose the precision (half, float, int16 or int8)
precision = 'int16'

# Load the data
f = open('../data/mnist.pkl', 'rb')
training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
f.close()

# Test how accurate the network has learned it's task
error = 0
testnet = caffe.Net(str('net_' + precision + '.prototxt'), caffe.TEST, weights='net_trained.caffemodel')
for k in range(0,len(validation_data[0])):
    testnet.blobs['mnist_image'].data[:] = np.reshape(validation_data[0][k],(784)).astype(float)/255.0
    testnet.forward()
    predicted_number = np.argmax(testnet.blobs['pred'].data[:])
    print('Predicted: '+str(predicted_number)+', actual: '+str(validation_data[1][k]))
    if not (predicted_number == validation_data[1][k]):
        error += 1
print('Errors: '+str(error)+' of '+str(len(validation_data[0]))+' ('+str(100.0-100.0*((float)(error)/(float)(len(validation_data[0]))))+'% accuracy)')
