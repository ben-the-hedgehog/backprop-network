__author__ = "Ben Pang"
__copyright__ = "2018, Ben Pang"

import numpy as np 

def precision_recall(confusion_matrix):
	result = []
	for i in range(confusion_matrix.shape[0]):
		precision = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
		recall = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
		result.append({'precision' : precision, 'recall' : recall})

	return result