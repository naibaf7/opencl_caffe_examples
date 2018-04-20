import sys
sys.path.append('..')
from caffe_examples_setup import *

# Choose the precision (half, float, int16 or int8).
precision = 'float'
batch_size = 256
do_training = False

# Load the data
f = open('../data/mnist.pkl', 'rb')
training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
f.close()

# Create a solver with a few typical parameters
# The solver will perform SGD on our data
solver_config = caffe.SolverParameter()
solver_config.train_net = 'net_' + precision + '_' + str(batch_size)  + '.prototxt'
solver_config.base_lr = 0.01
solver_config.momentum = 0.9
solver_config.weight_decay = 0.0005
solver_config.lr_policy = 'step'
solver_config.gamma = 0.1
solver_config.stepsize = 100000
solver_config.max_iter = 450000
solver_config.snapshot = 10000
solver_config.snapshot_prefix = 'net'
solver_config.display = 100

if do_training:
    # Do the training
    losses = []
    
    plt.ion()
    plot_obj, = plt.plot(losses)
    ax = plt.gca()
    plt.show()
    plt.pause(0.001)
    
    solver = caffe.get_solver(solver_config)
    for i in range(0, solver_config.max_iter):
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
testnet = caffe.Net(str('net_' + precision + '_' + str(batch_size)  + '.prototxt'), caffe.TEST, weights='bvlc_reference_caffenet.caffemodel')
# Enable quantizer observation
testnet.quant_mode = caffe.quantizer_mode.CAFFE_QUANT_OBSERVE
for k in range(0,10):
    testnet.forward()
    if (k % 100 == 0):
      print(k)
# Store the network parameters, including obtain quantizer information
testnet.save('net_trained.caffemodel')
print("Done.")


