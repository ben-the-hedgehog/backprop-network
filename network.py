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
			init_parameter = 1 / prev_outputs

			# initialize n x m weight matrix using uniform initialization, each row is a node's input weights
			weight_mtx = np.random.uniform(-init_parameter, init_parameter, size=(layers[i], prev_outputs)) 

			# n x 1 bias vector, each row is a node's bias
			bias_vector = np.zeros((layers[i], 1))	

			self.weights.append(weight_mtx)
			self.biases.append(bias_vector)

