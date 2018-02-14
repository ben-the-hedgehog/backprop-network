import numpy as np


class bp_network(object):

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

	def backpropogate(self, x, y):
		"""
		Apply the backpropogation algorithm to generate the deltas for weights
		and biases in each layer

		:param x:<np.arrray> 1 x m feature vector from dataset
		:param y:<double> numeric target
		"""
		# These will hold the gradient of the cost function with respect to
		# weights and biases
		del_w = [np.zeros(w.shape) for w in self.weights]
		del_b = [np.zeros(b.shape) for b in self.biases]

		# Forward pass through the network
		activation = x
		activations = [activation]
		net_inputs = []

		for w, b in zip(self.weights, self.biases):
			net_input = np.dot(w, activation) + b
			net_inputs.append(net_input)
			activation = sigmoid(net_input)
			activations.append(activation)

		# Error at output layer


	def cost_derivative(self, output_activations, y):
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
