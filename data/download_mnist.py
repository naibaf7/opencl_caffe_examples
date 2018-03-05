import sys
sys.path.append('..')
from caffe_examples_setup import *

# Download the MNIST data set, if not done already
if not os.path.isfile('../data/mnist.pkl.gz'):
    print('Downloading MNIST...')
    urllib.request.urlretrieve('https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz',
                    '../data/mnist.pkl.gz')

# Unzip the MNIST data set, if not done already
if not os.path.isfile('../data/mnist.pkl'):
    print('Unzipping MNIST...')
    input_file = gzip.open('../data/mnist.pkl.gz', 'rb')
    output_file = open('../data/mnist.pkl', 'wb')
    try:
        output_file.write(input_file.read())
    finally:
        input_file.close()
        output_file.close()

print('MNIST data available')

# Load the data
f = open('../data/mnist.pkl', 'rb')
training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
f.close()
plt.imshow(np.reshape(training_data[0][0],(28,28)))
plt.show()
