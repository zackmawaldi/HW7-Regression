import numpy as np
import matplotlib.pyplot as plt

# Base class for generic regressor
# (this is already complete!)
class BaseRegressor():

    def __init__(self, num_feats, learning_rate=0.01, tol=0.001, max_iter=100, batch_size=10):

        # Weights are randomly initialized
        self.W = np.random.randn(num_feats + 1).flatten()

        # Store hyperparameters
        self.lr = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_feats = num_feats

        # Define empty lists to store losses over training
        self.loss_hist_train = []
        self.loss_hist_val = []
    
    def make_prediction(self, X):
        return self.make_prediction(X)
    
    def loss_function(self, y_true, y_pred):
        return self.loss_function(y_true, y_pred)
        
    def calculate_gradient(self, y_true, X):
        return self.calculate_gradient(y_true, X)
    
    def train_model(self, X_train, y_train, X_val, y_val):

        # Padding data with vector of ones for bias term
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    
        # Defining intitial values for while loop
        prev_update_size = 1
        iteration = 1

        # Repeat until convergence or maximum iterations reached
        while prev_update_size > self.tol and iteration < self.max_iter:

            # Shuffling the training data for each epoch of training
            shuffle_arr = np.concatenate([X_train, np.expand_dims(y_train, 1)], axis=1)
            np.random.shuffle(shuffle_arr)
            X_train = shuffle_arr[:, :-1]
            y_train = shuffle_arr[:, -1].flatten()

            # Create batches
            num_batches = int(X_train.shape[0] / self.batch_size) + 1
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)

            # Create list to save the parameter update sizes for each batch
            update_sizes = []

            # Iterate through batches (one of these loops is one epoch of training)
            for X_train, y_train in zip(X_batch, y_batch):

                # Make prediction and calculate loss
                y_pred = self.make_prediction(X_train)
                train_loss = self.loss_function(y_train, y_pred)
                self.loss_hist_train.append(train_loss)

                # Update weights
                prev_W = self.W
                grad = self.calculate_gradient(y_train, X_train)
                new_W = prev_W - self.lr * grad 
                self.W = new_W

                # Save parameter update size
                update_sizes.append(np.abs(new_W - prev_W))

                # Compute validation loss
                val_loss = self.loss_function(y_val, self.make_prediction(X_val))
                self.loss_hist_val.append(val_loss)

            # Define step size as the average parameter update over the past epoch
            prev_update_size = np.mean(np.array(update_sizes))

            # Update iteration
            iteration += 1
    
    def plot_loss_history(self):

        # Make sure training has been run
        assert len(self.loss_hist_train) > 0, "Need to run training before plotting loss history."

        # Create plot
        fig, axs = plt.subplots(2, figsize=(8, 8))
        fig.suptitle('Loss History')
        axs[0].plot(np.arange(len(self.loss_hist_train)), self.loss_hist_train)
        axs[0].set_title('Training')
        axs[1].plot(np.arange(len(self.loss_hist_val)), self.loss_hist_val)
        axs[1].set_title('Validation')
        plt.xlabel('Steps')
        fig.tight_layout()
        plt.show()

    def reset_model(self):
        self.W = np.random.randn(self.num_feats + 1).flatten()
        self.loss_hist_train = []
        self.loss_hist_val = []
        
# Implement logistic regression as a subclass
class LogisticRegressor(BaseRegressor):

    def __init__(self, num_feats, learning_rate=0.01, tol=0.001, max_iter=100, batch_size=10):
        super().__init__(
            num_feats,
            learning_rate=learning_rate,
            tol=tol,
            max_iter=max_iter,
            batch_size=batch_size
        )
    

    def sigmoid(self, x):
        ''' maps x onto sigmoid function '''
        return 1 / (1 + np.exp(-x))


    def make_prediction(self, X) -> np.array:
        """
        Implement logistic function to get estimates (y_pred) for input X values. The logistic
        function is a transformation of the linear model into an "S-shaped" curve that can be used
        for binary classification.

        Arguments: 
            X (np.ndarray): Matrix of feature values.

        Returns: 
            The predicted labels (y_pred) for given X.
        """

                        # sigmoid( dot(X,W) + b ) # b is done via stub addition in the init
        prediction = self.sigmoid( np.dot(X, self.W) )

        return prediction



    
    def loss_function(self, y_true, y_pred) -> float:
        """
        Implement binary cross entropy loss, which assumes that the true labels are either
        0 or 1. (This can be extended to more than two classes, but here we have just two.)

        Arguments:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Returns: 
            The mean loss (a single number).
        """

        n = len( y_true )

        loss = -1/n * np.sum( y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return loss
        
    def calculate_gradient(self, y_true, X) -> np.ndarray:
        """
        Calculate the gradient of the loss function with respect to the given data. This
        will be used to update the weights during training.

        Arguments:
            y_true (np.array): True labels.
            X (np.ndarray): Matrix of feature values.

        Returns: 
            Vector of gradients.
        """
        # derivation of the gradient, which I will not
        # pretend to understand, is on page 16 and 22 of the following:
        # web.stanford.edu/~jurafsky/slp3/5.pdf

        n = len( y_true )

        y_pred = self.make_prediction(X)

        gradient = 1/n * np.dot(X.T, y_pred - y_true)

        return gradient