import sys
sys.path.append('..')
from caffe_examples_setup import *

# Choose the precision (half, float, int8 or int16).
precision = 'float'

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

# Create a solver with a few typical parameters
# The solver will perform SGD on our data
solver_config = caffe.SolverParameter()
solver_config.train_net = 'net_' + precision + '.prototxt'
solver_config.base_lr = 1.0
solver_config.momentum = 0.99
solver_config.weight_decay = 0.00005
solver_config.lr_policy = 'inv'
solver_config.gamma = 0.01
solver_config.power = 0.75
solver_config.max_iter = 2000
solver_config.snapshot = 500
solver_config.snapshot_prefix = 'net'
solver_config.type = 'Adam'
solver_config.display = 1

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
    k = random.randint(0,len(values_celsius_train)-1)
    # Load the sample into the network
    solver.net.blobs['celsius'].data[0] = values_celsius_train[k]
    solver.net.blobs['farenheit'].data[0] = values_farenheit_train[k]
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
error = []
testnet = caffe.Net(str('net_' + precision + '.prototxt'), caffe.TEST, weights='net_iter_'+str(solver_config.max_iter)+'.caffemodel')
# Enable quantizer observation
testnet.quant_mode = caffe.quantizer_mode.CAFFE_QUANT_OBSERVE
for c,f in zip(values_celsius_test,values_farenheit_test):
    testnet.blobs['celsius'].data[0] = c
    testnet.forward()
    predicted_f = testnet.blobs['output'].data[0,0]
    print('Cesius: '+str(c)+'°C, predicted: '+str(predicted_f)+' °F, actual: '+str(f)+' °F')
    error.append(abs(f-predicted_f))
    
print('Average error: '+str(np.array(error).mean())+' °F')

# Store the network parameters, including obtain quantizer information
testnet.save('net_trained.caffemodel')
print("Done.")


