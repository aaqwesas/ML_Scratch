# Import necessary libraries
import pandas as pd                # For data manipulation and analysis
import numpy as np               # For numerical operations with arrays and matrices
import matplotlib.pyplot as plt  # For creating plots and visualizations
from matplotlib.colors import ListedColormap  # For creating customized colormaps for plots

# ===============================
# Data Loading and Preprocessing
# ===============================

# Load the Iris dataset from the UCI Machine Learning Repository.
# The dataset does not have a header row, so we set header=None.
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header=None)

# Extract the class labels (target values) for the first 100 samples.
# Column index 4 contains the class names (i.e., 'Iris-setosa' or 'Iris-versicolor').
y = df.iloc[0:100, 4]

# Convert the class labels to a binary representation:
# Label 'Iris-setosa' as -1 and all others (i.e., 'Iris-versicolor') as 1.
y = np.where(y == 'Iris-setosa', -1, 1)

# Extract two features (columns) from the data: column 0 and column 2.
# In this example, we use sepal length (column 0) and petal length (column 2)
X = df.iloc[0:100, [0, 2]].values

# ======================================================
# Adaline with Batch Gradient Descent (Adaptive Linear Neuron)
# ======================================================

class AdalineGD(object):
    """
    ADAptive LInear NEuron (Adaline) classifier using Batch Gradient Descent.

    Parameters:
    -----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Number of passes (epochs) over the training dataset.
    random_state : int
        Seed for the random number generator for reproducibility.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta                # Learning rate
        self.n_iter = n_iter          # Number of epochs
        self.random_state = random_state  # Seed for initializing weights

    def fit(self, X, y):
        """
        Fit training data using batch gradient descent.

        Parameters:
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            Training data.
        y : array-like, shape = [n_samples]
            Target values.

        Returns:
        --------
        self : object
        """
        # Initialize a random number generator with the specified seed.
        rgen = np.random.RandomState(self.random_state)
        # Initialize weights to small random numbers.
        # The weight vector includes one bias unit (w_0) and one weight for each feature.
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        # List to store the cost value in each epoch (cost = sum-of-squared errors)
        self.cost_ = []

        # Begin the learning process (looping over epochs)
        for i in range(self.n_iter):
            # Calculate the net input (i.e., weighted sum plus bias) for all training samples.
            net_input = self.net_input(X)
            # Activation function is identity function for Adaline; it's linear.
            output = self.activation(net_input)
            # Compute the errors by subtracting the activation output from the true labels.
            errors = (y - output)
            # Update the weights (excluding bias) using the gradient descent update rule.
            self.w_[1:] += self.eta * X.T.dot(errors)
            # Update the bias term; note that the sum over errors is applied.
            self.w_[0] += self.eta * errors.sum()
            # Calculate the cost (sum-of-squared errors divided by 2) for monitoring.
            cost = (errors**2).sum() / 2.0
            # Append the cost value for this epoch.
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """
        Calculate the net input.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input data.

        Returns:
        --------
        float or array-like: The net input value computed as dot product of X and weights plus bias.
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """
        Compute the linear activation.

        For Adaline, activation is simply the identity function.
        
        Parameters:
        -----------
        X : array-like
            Net input.

        Returns:
        --------
        array-like: Activation output (same as input for Adaline).
        """
        return X

    def predict(self, X):
        """
        Return class label after applying the unit step function.

        If the activation is >= 0 then class label is set to 1, else -1.

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input data.

        Returns:
        --------
        array-like: Predicted class labels.
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

# ======================================================
# Adaline with Stochastic Gradient Descent (SGD)
# ======================================================

class AdalineSGD(object):
    """
    ADAptive LInear NEuron (Adaline) classifier using Stochastic Gradient Descent (SGD).

    Parameters:
    -----------
    eta : float
        Learning rate.
    n_iter : int
        Passes over the training dataset.
    shuffle : bool (default: True)
        Shuffles training data every epoch to prevent cycles.
    random_state : int or None
        Seed for the random number generator for reproducibility.
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta                    # Learning rate
        self.n_iter = n_iter              # Number of epochs
        self.w_initialized = False        # Flag to check if weights have been initialized
        self.shuffle = shuffle            # Option to shuffle training data at each epoch
        self.random_state = random_state  # Random seed for reproducibility

    def fit(self, X, y):
        """
        Fit training data using stochastic gradient descent.

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
        y : array-like, shape = [n_samples]
            Target values.

        Returns:
        --------
        self : object
        """
        # Initialize weights for the given number of features.
        self._initialize_weights(X.shape[1])
        # Initialize list to store cost in each epoch.
        self.cost_ = []

        # Iterate over the specified number of epochs.
        for i in range(self.n_iter):
            # Shuffle training data if the flag is set to True.
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []  # List to record cost for each individual training example in the epoch.
            # Loop over each training sample.
            for xi, target in zip(X, y):
                # Update weights using the current training sample and record the cost.
                cost.append(self._update_weights(xi, target))
            # Calculate the average cost for this epoch.
            avg_cost = sum(cost) / len(y)
            # Append the average cost to the cost history.
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """
        Update weights without reinitializing them.
        Allows for online/incremental learning.

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input data.
        y : array-like, shape = [n_samples]
            Target values.

        Returns:
        --------
        self : object
        """
        # Check if weights are already initialized; if not, do so.
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        # If y contains more than one sample, update weights for each sample.
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            # For a single sample, update the weights directly.
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """
        Shuffle the training data to avoid cycles in the weight updates.

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
        y : array-like, shape = [n_samples]
            Target values.
        
        Returns:
        --------
        X, y : Shuffled training data and target values.
        """
        r = self.rgen.permutation(len(y))  # Generate a permutation of indices.
        return X[r], y[r]

    def _initialize_weights(self, m):
        """
        Initialize weights to small random numbers.

        Parameters:
        -----------
        m : int
            Number of features.
        """
        # Create a random number generator using the specified seed.
        self.rgen = np.random.RandomState(self.random_state)
        # Initialize weights including the bias weight.
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """
        Apply weight update for a single training sample.

        Parameters:
        -----------
        xi : array-like, shape = [n_features]
            Input feature vector.
        target : float
            True class label.

        Returns:
        --------
        cost : float
            The cost (error) for the given training sample.
        """
        # Compute the net input and activation for the current sample.
        output = self.activation(self.net_input(xi))
        # Calculate the error by comparing the target with the output.
        error = (target - output)
        # Update the weights for the features.
        self.w_[1:] += self.eta * xi.dot(error)
        # Update the bias weight.
        self.w_[0] += self.eta * error
        # Calculate the cost (using squared error) for this sample.
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """
        Compute the net input, which is the weighted sum of the input features and bias.

        Parameters:
        -----------
        X : array-like, shape = [n_features] or [n_samples, n_features]
            Input data.

        Returns:
        --------
        float or array-like: The net input value.
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """
        Compute linear activation (identity function).

        Parameters:
        -----------
        X : array-like
            Net input.

        Returns:
        --------
        array-like: Activation output (same as the net input).
        """
        return X

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Uses a unit step function over the activation function output.

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input data.

        Returns:
        --------
        array-like: Predicted class labels.
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

# ======================================================
# Utility Function: Plotting Decision Regions
# ======================================================

def plot_decision_regions(X, y, classifier, resolution=0.01):
    """
    Plot the decision regions (boundaries) for a trained classifier.

    Parameters:
    -----------
    X : array-like, shape = [n_samples, n_features]
        Feature data.
    y : array-like, shape = [n_samples]
        True class labels.
    classifier : object
        Classifier which supports the .predict() method.
    resolution : float
        Step size for the mesh grid.
    """
    # Define marker types and colors for different class labels.
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # Create a custom colormap using only the number of colors needed for unique classes.
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Determine the minimum and maximum values for the first feature.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # Determine the minimum and maximum values for the second feature.
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid with the given resolution over the feature space.
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # Predict the class labels for each point in the grid.
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # Reshape the result to match the shape of the mesh grid.
    Z = Z.reshape(xx1.shape)

    # Draw the decision surface by plotting a filled contour plot.
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot the original data points on top of the decision regions.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

# ======================================================
# Main Function: Training and Visualization
# ======================================================

def main():
    """
    Main function to execute training and visualizing the Adaline models.
    This function demonstrates the effect of various learning rates and the performance
    of both the batch gradient descent and stochastic gradient descent implementations.
    """

    # ---------------------------
    # Adaline with Batch Gradient Descent
    # ---------------------------

    # Train Adaline using batch gradient descent with a smaller learning rate (0.0001) and 30 epochs.
    ada = AdalineGD(n_iter=30, eta=0.0001).fit(X, y)
    # The cost history is stored in ada.cost_ (not directly used here).

    # Create a side-by-side plot to compare impact of two different learning rates.
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Train with a relatively higher learning rate (0.01) for 10 epochs.
    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    # Plot the logarithm of the sum-of-squared errors for each epoch to observe convergence.
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    # Train with a smaller learning rate (0.0001) for 10 epochs.
    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    # Plot the actual sum-of-squared errors per epoch.
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    # Display the two plots.
    plt.show()

    # ---------------------------
    # Standardization of Features
    # ---------------------------
    # Copy X into X_std for standardization.
    X_std = np.copy(X)
    # Standardize the first feature: subtract the mean and divide by the standard deviation.
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    # Standardize the second feature.
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # ---------------------------
    # Adaline with Batch Gradient Descent on Standardized Data
    # ---------------------------
    # Train Adaline on standardized features for better convergence.
    ada = AdalineGD(n_iter=20, eta=0.01)
    ada.fit(X_std, y)
    # Plot decision regions using the trained model.
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # Plot the sum-of-squared error cost over epochs to analyze the convergence.
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Adaline with Stochastic Gradient Descent on Standardized Data
    # ---------------------------
    # Train the Adaline using SGD with 15 epochs.
    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)
    # Plot decision regions for the classifier trained with SGD.
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # Plot the average cost per epoch to inspect the learning progress.
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.tight_layout()
    plt.show()

# Ensures that the main function is called when the script is executed.
if __name__ == '__main__':
    main()