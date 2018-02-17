import numpy as np


class bp_classifer(object):

	def __init__(self, layers, learning_rate, momentum):
		"""
		Initializes the feedforward neural network

		:param layers:<list> layer specification for the FF network in list form (eg. [5 3 3 2])
		:param learning_rate:<double> learning rate of the network (C)
		:param momentum:<double> momentum parameter (alpha)
		"""
		self.c = learning_rate
		self.alpha = momentum
		self.num_inputs = layers[0]
		self.num_outputs = layers[-1]
		self.num_layers = len(layers)

		self.weights = []
		self.biases = []

		for i in range(1, self.num_layers):
			prev_outputs = layers[i-1]
			init_parameter = 1.0 / prev_outputs

			# initialize n x m weight matrix using uniform initialization, each row is a node's input weights
			weight_mtx = np.random.uniform(-init_parameter, init_parameter, size=(layers[i], prev_outputs)) 

			# n x 1 bias vector, each row is a node's bias
			bias_vector = np.zeros((layers[i], 1))	

			self.weights.append(weight_mtx)
			self.biases.append(bias_vector)

	def one_hot(self, d):
		"""transforms a numeric label into one-hot column vector"""
		y = np.zeros((self.num_outputs, 1))
		y[d] = 1
		return y

	def train(self, train_X, train_D, val_X=None, val_D=None, num_epochs=1):

		train_Y = list(map(self.one_hot, train_D))
		val_Y = list(map(self.one_hot, val_D)) if val_D else None
		error = []

		for epoch in range(num_epochs):

			for x, y in zip(train_X, train_Y):
				self.update_SGD(x.reshape(1, self.num_inputs), y)


	def update_SGD(self, x, y):
		"""
		Applies a single step of stochastic gradient descent

		:param x:<np.arrray> 1 x m feature vector from dataset
		:param y:<np.array> one-hot target vector indicating desired label
		"""

		del_w, del_b = self.backpropogate(x, y)
		self.weights = [w - self.c * dw + self.alpha * w for w, dw in zip(self.weights, del_w)]
		self.biases = [b - self.c * db + self.alpha * b for b, db in zip(self.biases, del_b)]

		return None

	def backpropogate(self, x, y):
		"""
		Apply the backpropogation algorithm to generate the deltas for weights
		and biases in each layer

		:param x:<np.arrray> 1 x m feature vector from dataset
		:param y:<np.array> one-hot target vector indicating desired label
		"""

		# These will hold the gradient of the cost function with respect to
		# weights and biases
		del_w = [np.zeros(w.shape) for w in self.weights]
		del_b = [np.zeros(b.shape) for b in self.biases]

		# Forward pass through the network
		activation = x.T
		activations = [activation]
		net_inputs = []

		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation) + b
			net_inputs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		# Error at output layer
		output = self.one_hot(np.argmax(activations[-1]))
		delta = self.cost_derivative(output, y) * sigmoid_prime(net_inputs[-1]) #dC/dz at output
		del_b[-1] = delta #dC/db = dC/dz
		del_w[-1] = np.dot(delta, activations[-2].T)

		# propogate error backwards layer by layer
		for l in range(2, self.num_layers):
			z = net_inputs[-l]
			delta = np.dot(self.weights[-l+1].T, delta) * sigmoid_prime(z)
			del_b[-l] = delta
			del_w[-l] = np.dot(delta, activations[-l-1].T)

		return (del_w, del_b)

	@staticmethod
	def cost_function(output_activations, y):
		"""use 1/2 of the square error as the cost function"""
		return (1/2) * (y - output_activations) ** 2

	@staticmethod
	def cost_derivative(output_activations, y):
		"""derivative of the L2 cost function with respect to network output"""
		return (output_activations - y)


def sigmoid(z):
	"""
	The sigmoid function
	"""
	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
	"""
	derivative of the sigmoid function
	"""
	return sigmoid(z) * (1 - sigmoid(z))
