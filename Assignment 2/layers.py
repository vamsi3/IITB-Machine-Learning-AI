import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE

		out_activation = sigmoid(X @ self.weights + self.biases)
		self.data = out_activation
		return out_activation

		# raise NotImplementedError
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE

		out_activation = self.data

		del_in = delta * out_activation * (1 - out_activation)
		new_delta = del_in @ self.weights.T

		self.biases  -= lr * del_in.sum(axis=0)
		self.weights -= lr * activation_prev.T @ del_in
		return new_delta

		# raise NotImplementedError
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

		### ADDED BY VAMSI - This can be done in self.forwardpass() also but placing it here is more time-efficient.
		## k, i, j are indices of X [(depth, row, col)] accessed in row-major order of each convolution stacked in the order of row-major convolutions
		## (Slightly difficult to describe, but very intuitive!)

		# Here i finally is a (in_depth * filter_row * filter_col) x (out_row * out_col)
		i_filter_indices = np.tile(np.repeat(np.arange(self.filter_row), self.filter_col), self.in_depth).reshape(-1, 1).astype(int)
		i_convolve_indices = self.stride * np.repeat(np.arange(self.out_row), self.out_col).reshape(1, -1).astype(int)
		self.i = i_filter_indices + i_convolve_indices # Using broadcasting to our advantage

		# Similarly for j. Here also we get a (in_depth * filter_row * filter_col) x (out_row * out_col)
		j_filter_indices = np.tile(np.arange(self.filter_col), self.filter_row * self.in_depth).reshape(-1, 1).astype(int)
		j_convolve_indices = self.stride * np.tile(np.arange(self.out_col), self.out_row).reshape(1, -1).astype(int)
		self.j = j_filter_indices + j_convolve_indices

		# For k, we save memory by using broadcasting to our advantage
		# Here k is a (in_depth * filter_row * filter_col) x 1 matrix, which is OK since broadcasting while using k, i, j together will give proper full k matrix (alteast for intuition)
		# This could be done here, since k has same columns if written as a full matrix like i, j on a paper.
		self.k = np.repeat(np.arange(self.in_depth), self.filter_row * self.filter_col).reshape(-1, 1).astype(int)


	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_channels[0] X self.in_channels[1] X self.in_channels[2]]
		# OUTPUT activation matrix		:[n X self.outputsize[0] X self.outputsize[1] X self.numfilters]

		###############################################
		# TASK 1 - YOUR CODE HERE

		## The basic idea I'm doing here is that since weights are shared, each input value interacts with multiple weights to possibly contribute to different output activations
		## So, we duplicate these X entries (Yes! indeed requires quite higher memory), in a very specific order so that convolution boils down to X_flattened and W_flattened interacting exactly same as a linear layer
		## After working on flattened space, we reshape and go back properly to our actual depth x row x col spaces.

		X_flattened = X[:, self.k, self.i, self.j].transpose(1, 2, 0).reshape(self.in_depth * self.filter_row * self.filter_col, -1)
		W_flattened = self.weights.reshape(self.out_depth, -1)
		out_activation = sigmoid(W_flattened @ X_flattened + self.biases.reshape(-1, 1))

		# We'll go back to original shapes now!
		out_activation = out_activation.reshape(self.out_depth, self.out_row, self.out_col, n).transpose(3, 0, 1, 2)
		self.data = (out_activation, X_flattened)
		return out_activation

		# raise NotImplementedError
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE

		## Just similar ot forward propagation, working in flattened space and going back to depth x row x col space. This time getting the new_delta is quite tricky.
		## We use the np.add.at() function which we use in such a way that same (k, i, j) tuples that contributed more than once, their corresponding contributing gradients gets added up.

		# X_flattened = activation_prev[:, self.k, self.i, self.j].transpose(1, 2, 0).reshape(self.in_depth * self.filter_row * self.filter_col, -1)
		out_activation, X_flattened = self.data # X_flattened can also be obtained as above, but storing it in self.data for reuse is more time efficient.
		del_in = delta * out_activation * (1 - out_activation)
		
		del_in_flattened = del_in.transpose(1, 2, 3, 0).reshape(self.out_depth, -1)
		W_flattened = self.weights.reshape(self.out_depth, -1)
		new_delta_flattened = W_flattened.T @ del_in_flattened
		new_delta_flattened = new_delta_flattened.reshape(self.in_depth * self.filter_row * self.filter_col, -1, n).transpose(2, 0, 1)

		# We'll go back to original shapes now!
		new_delta = np.zeros((n, self.in_depth, self.in_row, self.in_col), dtype=new_delta_flattened.dtype) # Ensure same type so that np.add.at() doesn't cause any type troubles.
		np.add.at(new_delta, (slice(None), self.k, self.i, self.j), new_delta_flattened) # Described above

		self.biases -= lr * np.sum(del_in, axis=(0, 2, 3))
		self.weights -= lr * (del_in_flattened @ X_flattened.T).reshape(self.weights.shape)
		return new_delta

		# raise NotImplementedError
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_channels[0] X self.in_channels[1] X self.in_channels[2]]
		# OUTPUT activation matrix		:[n X self.outputsize[0] X self.outputsize[1] X self.in_channels[2]]

		###############################################
		# TASK 1 - YOUR CODE HERE

		return X.reshape(n, self.in_depth, self.in_row // self.filter_row, self.filter_row, self.in_col // self.filter_col, self.filter_col).mean(axis=(3, 5))

		# raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE

		new_delta = np.repeat(delta, self.filter_col).reshape(n, self.out_depth, self.out_row, self.in_col) / self.filter_col # stretch along cols to get out_row x in_col matrix
		new_delta = np.tile(new_delta, self.filter_row).reshape(n, self.in_depth, self.in_row, self.in_col) / self.filter_row # stretch along rows to get in_row x in_col matrix
		return new_delta

		# raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))