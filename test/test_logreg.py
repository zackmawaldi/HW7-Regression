"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np

import regression

def test_prediction():
	''' prediciton check via known W and X '''

	# from a known good W and X
	W = np.array([-0.87076156, -0.30572202, -0.339589, -0.11416402, -0.83947357, 0.05316974, -0.15280408])
	X = np.array([[1.39734081, 1.55586084, 0.13033579, 0, 0, 53, 1]])

	ground_truth = 1 / (1 + np.exp(-np.dot(X, W)) )

	reg = regression.LogisticRegressor(num_feats=6)
	reg.W = W

	assert reg.make_prediction(X) == ground_truth
	

def test_loss_function():
	''' loss check via y_true and y_pred '''

	y_true = np.array([0, 1])
	y_pred = np.array([0.2, 0.9])

	ground_truth = -1/2 * np.sum( y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) 

	reg = regression.LogisticRegressor(num_feats=6, batch_size=2)
	
	assert reg.loss_function(y_true, y_pred) == ground_truth

def test_gradient():
	''' gradient check via known W, X, and y_true '''

	W = np.array([-0.87076156, -0.30572202, -0.339589, -0.11416402, -0.83947357, 0.05316974, -0.15280408])
	X = np.array([[1.39734081, 1.55586084, 0.13033579, 0, 0, 53, 1]])
	y_true = np.array([1])

	reg = regression.LogisticRegressor(num_feats=6)
	reg.W = W

	pred_grad = list( reg.calculate_gradient(y_true, X) )

	ground_truth = [-0.39576196, -0.44065881, -0.03691436, 0 ,0 , -15.01092922,  -0.28322508]

	for pred, ground in zip( pred_grad, ground_truth ):
		assert round( pred, 5 ) == round( ground, 5 )



def test_training():
	''' training check via comparing post-training wights with initial '''

	X_train, X_test, Y_train, Y_test = regression.utils.loadDataset(split_percent=0.9)

	num_feats = X_train.shape[-1]
	reg = regression.LogisticRegressor(num_feats)
	
	random_W = np.random.randn(num_feats + 1).flatten()
	reg.W = random_W
	
	reg.train_model(X_train, Y_train, X_test, Y_test)

	assert not np.array_equal( reg.W , random_W )