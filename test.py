import network
import numpy as np 


def init_test():
	net = network.bp_classifer([3, 2, 3, 1], 0.1, 0.01)
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
	net = network.bp_classifer([3, 2, 3, 3], 0.1, 0.01)
	x = np.array([1, 3, 0]).reshape(1, 3)
	#y = np.array([2]).reshape(1, 1)
	y = net.one_hot(2)
	dw, db = net.backpropogate(x, y)
	weight_shapes = [w.shape for w in dw]
	assert weight_shapes == [(2, 3), (3, 2), (3, 3)]
	bias_shapes = [b.shape for b in db]
	assert bias_shapes == [(2, 1), (3, 1), (3, 1)]
	print("backprop_test passed!\n")
	pass

def update_test():
	net = network.bp_classifer([3, 2, 3], 0.1, 0.01)
	x = np.array([1, 3, 0]).reshape(1, 3)
	#y = np.array([0]).reshape(1, 1)
	y = net.one_hot(0)
	w = net.weights.copy()
	b = net.biases.copy()
	net.update_SGD(x, y)
	for i in range(len(w)):
		assert net.weights[i].shape == w[i].shape
		assert net.biases[i].shape == b[i].shape
	print("update_test passed!\n")
	pass

def train_test():
	net = network.bp_classifer([3, 2, 3], 0.1, 0.01)
	print("BEFORE TRAINING\n")
	for w, b in zip(net.weights, net.biases):
		print(w)
		print(b)

	train_X = np.random.randint(10, size=(10, 3))
	train_D = np.random.randint(3, size=(10, 1))
	net.train(train_X, train_D)

	print("AFTER TRAINING\n")
	for w, b in zip(net.weights, net.biases):
		print(w)
		print(b)

	print("train_test passed!\n")
	pass

def validation_test():
	net = network.bp_classifer([3, 2, 3], 0.1, 0.01)
	train_X = np.random.randint(10, size=(10, 3))
	train_D = np.random.randint(3, size=(10, 1))
	val_X = np.random.randint(10, size=(5, 3))
	val_D = np.random.randint(3, size=(5, 1))
	n_epochs = 5

	acc = net.train(train_X, train_D, val_X, val_D, num_epochs=n_epochs)
	for epoch in range(n_epochs):
		print(f'epoch {epoch + 1}: {acc[epoch]}')

	print("validation_test passed!\n")
	pass

if __name__ == '__main__':
	init_test()
	backprop_test()
	update_test()
	train_test()
	validation_test()