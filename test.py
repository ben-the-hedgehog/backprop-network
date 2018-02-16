import network
import numpy as np 


def init_test():
	net = network.bp_network([3, 2, 3, 1], 0.1, 0.01)
	assert net.num_layers == 4
	assert net.num_inputs == 3
	assert net.num_outputs == 1
	weight_shapes = [w.shape for w in net.weights]
	assert weight_shapes == [(2, 3), (3, 2), (1, 3)]
	bias_shapes = [b.shape for b in net.biases]
	assert bias_shapes == [(2, 1), (3, 1), (1, 1)]
	print("init_test passed!\n")
	pass

def backprop_test():
	net = network.bp_network([3, 2, 3, 1], 0.1, 0.01)
	x = np.array([1, 3, 0]).reshape(1, 3)
	y = np.array([6]).reshape(1, 1)
	dw, db = net.backpropogate(x, y)
	weight_shapes = [w.shape for w in dw]
	assert weight_shapes == [(2, 3), (3, 2), (1, 3)]
	bias_shapes = [b.shape for b in db]
	assert bias_shapes == [(2, 1), (3, 1), (1, 1)]
	print("backprop_test passed!\n")
	pass

def update_test():
	net = network.bp_network([3, 2, 3, 1], 0.1, 0.01)
	x = np.array([1, 3, 0]).reshape(1, 3)
	y = np.array([6]).reshape(1, 1)
	w = net.weights.copy()
	b = net.biases.copy()
	net.update_SGD(x, y)
	for i in range(len(w)):
		assert np.array_equal(net.weights[i], w[i]) is False
		assert np.array_equal(net.biases[i], b[i]) is False
	print("update_test passed!\n")
	pass

if __name__ == '__main__':
	init_test()
	backprop_test()
	update_test()