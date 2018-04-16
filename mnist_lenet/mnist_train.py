import sys
sys.path.append('..')
from caffe_examples_setup import *

# Choose the precision (half, float, int16 or int8).
precision = 'float'

# Load the data
f = open('../data/mnist.pkl', 'rb')
training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
f.close()

# Create a solver with a few typical parameters
# The solver will perform SGD on our data
solver_config = caffe.SolverParameter()
solver_config.train_net = 'net_' + precision + '.prototxt'
solver_config.base_lr = 0.001
solver_config.momentum = 0.95
solver_config.weight_decay = 0.00005
solver_config.lr_policy = 'inv'
solver_config.gamma = 0.001
solver_config.power = 0.75
solver_config.max_iter = 8000
solver_config.snapshot = 2000
solver_config.snapshot_prefix = 'net'
solver_config.type = 'Adam'
solver_config.display = 100

# Do the training
losses = []

plt.ion()
plot_obj, = plt.plot(losses)
ax = plt.gca()
plt.show()
plt.pause(0.001)

solver = caffe.get_solver(solver_config)
for i in range(0, solver_config.max_iter):
    # Pick a random sample for training
    k = random.randint(0,len(training_data[0])-1)
    # Load the sample into the network
    solver.net.blobs['mnist_image'].data[:] = np.reshape(training_data[0][k],(28,28)).astype(float)/255.0
    solver.net.blobs['label'].data[0] = training_data[1][k]
    # Train one step
    loss = solver.step(1)
    # Display the learning progress every 20 steps
    if (i % 100 == 0):
        losses.append(loss)
        plot_obj.set_data(range(0,len(losses)), losses)
        ax.relim()
        ax.autoscale_view(True, True, True)
        plt.draw()
        plt.pause(0.001)


# Run a few test steps to observe the value ranges (for quantization)
error = 0
testnet = caffe.Net(str('net_' + precision + '.prototxt'), caffe.TEST, weights='net_iter_'+str(solver_config.max_iter)+'.caffemodel')
# Enable quantizer observation
testnet.quant_mode = caffe.quantizer_mode.CAFFE_QUANT_OBSERVE
for k in range(0,len(validation_data[0])):
    testnet.blobs['mnist_image'].data[:] = np.reshape(validation_data[0][k],(28, 28)).astype(float)/255.0
    testnet.forward()
    if (k % 100 == 0):
      print(k)
# Store the network parameters, including obtain quantizer information
testnet.save('net_trained.caffemodel')
print("Done.")


