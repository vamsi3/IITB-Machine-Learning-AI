import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''

	num_data_points, num_features = X.shape
	X_preprocessed = np.ones([num_data_points, 1])
	for i in range(1, num_features):
		col = X[:, i].reshape(-1, 1)
		if isinstance(col[0][0], str):
			col = one_hot_encode(col, list(np.unique(col)))
		else:
			col = col.astype(np.float64)
			mean, std = col.mean(0), col.std(0)
			col = np.divide(col - mean, std, where=(std != 0))
		X_preprocessed = np.concatenate((X_preprocessed, col), axis=1)
	return X_preprocessed, Y.astype(np.float64)

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''

	return 2 * (_lambda * W - X.T @ (Y - X @ W))

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''

	num_data_points, num_features = X.shape
	W = np.random.randn(num_features, 1) / np.sqrt(num_features) # Xavier Initialization
	for epoch in range(max_iter):
		grad_W = grad_ridge(W, X, Y, _lambda)
		W -= lr * grad_W
		if np.linalg.norm(grad_W) <= epsilon:
			break
	return W

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''

	num_data_points, num_features = X.shape
	split_X, split_Y = np.array_split(X, k), np.array_split(Y, k)
	average_sse_values = np.zeros(len(lambdas))
	for i in range(k):
		X_train, X_test = np.vstack(split_X[:i] + split_X[i+1:]), split_X[i]
		Y_train, Y_test = np.vstack(split_Y[:i] + split_Y[i+1:]), split_Y[i]
		for j, _lambda in enumerate(lambdas):
			W = algo(X_train, Y_train, _lambda)
			average_sse_values[j] += sse(X_test, Y_test, W)
	average_sse_values /= k
	return average_sse_values.tolist()

def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''

	_lambda /= 2
	num_features = np.size(X, 1)
	W = np.random.randn(num_features, 1) / np.sqrt(num_features) # Xavier Initialization
	memo_X_transpose_X, memo_X_transpose_Y = X.T @ X, X.T @ Y
	for epoch in range(max_iter):
		for k in range(num_features):
			X_k = X[:, k].reshape(-1, 1)
			tmp = memo_X_transpose_Y[k] - memo_X_transpose_X[k] @ W + memo_X_transpose_X[k][k] * W[k]
			W[k] = (tmp - np.sign(tmp - _lambda) * _lambda) / memo_X_transpose_X[k][k] if abs(tmp) > _lambda else 0
	return W

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	lambdas = np.linspace(0, 100, num=50, endpoint=False).tolist() # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, ridge_grad_descent)
	plot_kfold(lambdas, scores)

	lambdas = np.linspace(2*10**5, 6*10**5, num=40, endpoint=False).tolist() # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, coord_grad_descent)
	plot_kfold(lambdas, scores)