import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Perceptron:
    """
    Perceptron classifier.

    Parameters:
    ------------
    learning_rate : float, default=0.01
        Learning rate (between 0.0 and 1.0).
    iterations : int, default=50
        Number of passes over the training dataset.
    random_state : int, default=1
        Random number seed for weight initialization.

    Attributes:
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in each epoch.
    """

    def __init__(self, learning_rate=0.01, iterations=50, random_state=1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data.

        Parameters:
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors.
        y : array-like, shape = [n_samples]
            Target values (1 or -1).

        Returns:
        --------
        self : object
        """
        # Initialize weights
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        # Training loop
        for idx in range(self.iterations):
            errors = 0
            # Shuffle the data
            indices = np.arange(X.shape[0])
            rgen.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for xi, target in zip(X_shuffled, y_shuffled):
                prediction = self.predict(xi)
                update = self.learning_rate * (target - prediction)
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            print(f'Epoch {idx+1}/{self.iterations}, Misclassifications: {errors}')
            if errors == 0:
                print(f'Training converged after {idx+1} epochs.')
                break
        return self

    def net_input(self, X):
        """Calculate net input (weighted sum)."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def get_weights(self):
        """Return weights and bias."""
        return self.w_[1:], self.w_[0]

    def accuracy(self, X, y):
        """Calculate the accuracy of the model."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    Plot the decision regions of a classifier.

    This function visualizes the decision boundaries of a classifier on a 2D feature space.
    It creates a colored contour plot showing the decision regions and scatter plots
    of the input data points.

    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        The input samples. It is assumed to have exactly two features.
    y : array-like, shape (n_samples,)
        The target values (class labels) as integers.
    classifier : object
        A fitted classifier object that has a 'predict' method.
    resolution : float, optional (default=0.02)
        The step size of the mesh grid used for plotting.

    Returns:
    --------
    None
        This function does not return any value. It creates a plot using matplotlib.
    """
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, 
                    c=colors[idx],  marker=markers[idx],
                    label=cl, edgecolor='black')

def plot_training_errors(perceptron):
    plt.figure()
    plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.title('Perceptron - Training Errors')
    plt.grid(True)
    plt.show()

def main():
    # Load Iris dataset
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    
    # Prepare target values
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    
    # Extract features: sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values
    
    # Feature scaling
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Initialize Perceptron
    ppn = Perceptron(learning_rate=0.1, iterations=100, random_state=1)
    
    # Train Perceptron
    ppn.fit(X_std, y)
    
    # Plot decision regions
    plot_decision_regions(X_std, y, classifier=ppn)
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.legend(loc='upper left')
    plt.title('Perceptron - Decision Regions')
    plt.show()
    
    # Plot training errors
    plot_training_errors(ppn)
    
    # Calculate and print accuracy
    accuracy = ppn.accuracy(X_std, y)
    print(f'Perceptron classification accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()