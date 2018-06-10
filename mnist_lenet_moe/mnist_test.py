import sys
sys.path.append('..')
from caffe_examples_setup import *

# Choose the precision (half, float, int16 or int8)
precision = 'float'
batch_size = 128

# Load the data
f = open('../data/mnist.pkl', 'rb')
training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
f.close()

# Test how accurate the network has learned it's task
error = 0
testnet = caffe.Net(str('net_' + precision + '.prototxt'), caffe.TEST, weights='net_trained.caffemodel')
for k in range(0, len(validation_data[0]), batch_size):
    for b in range(0, batch_size):
      testnet.blobs['mnist_image'].data[b,:] = np.reshape(validation_data[0][k+b],(28, 28)).astype(float)/255.0
    testnet.forward()
    for b in range(0, batch_size):
      predicted_number = np.argmax(testnet.blobs['pred'].data[b,:])
      print('Predicted: '+str(predicted_number)+', actual: '+str(validation_data[1][k+b]))
      if not (predicted_number == validation_data[1][k+b]):
          error += 1
print('Errors: '+str(error)+' of '+str(len(validation_data[0]))+' ('+str(100.0-100.0*((float)(error)/(float)(len(validation_data[0]))))+'% accuracy)')
