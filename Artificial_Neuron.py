import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, 
                 weight_init='random', bias_init='zero', bias_constant=0.0):
        """
        Initialize the Perceptron.

        Parameters:
        -----------
        input_size : int
            Number of input features.
        learning_rate : float, default=0.1
            Learning rate for weight updates.
        weight_init : str, default='random'
            Method to initialize weights ('zero', 'random', 'xavier', 'he', 'lecun').
        bias_init : str, default='zero'
            Method to initialize bias ('zero', 'random', 'constant').
        bias_constant : float, default=0.0
            Constant value to set bias if bias_init='constant'.
        """
        self.learning_rate = learning_rate
        
        # Initialize weights based on the specified method
        if weight_init == 'zero':
            self.weights = np.zeros(input_size)
        elif weight_init == 'random':
            self.weights = np.random.randn(input_size) * 0.01  # Small random numbers
        elif weight_init == 'xavier':
            limit = np.sqrt(6 / input_size)
            self.weights = np.random.uniform(-limit, limit, size=input_size)
        elif weight_init == 'he':
            std = np.sqrt(2 / input_size)
            self.weights = np.random.normal(0, std, size=input_size)
        elif weight_init == 'lecun':
            std = np.sqrt(1 / input_size)
            self.weights = np.random.normal(0, std, size=input_size)
        else:
            raise ValueError("Unsupported weight_init method")
        
        # Initialize bias
        if bias_init == 'zero':
            self.bias = 0.0
        elif bias_init == 'random':
            self.bias = np.random.randn() * 0.01  # Small random number
        elif bias_init == 'constant':
            self.bias = bias_constant
        else:
            raise ValueError("bias_init must be 'zero', 'random', or 'constant'")
    
    def activation(self, x):
        """Heaviside step function."""
        return 1 if x >= 0 else -1  # Changed to output -1 for consistency with labels

    def predict(self, inputs):
        """Compute the neuron's output."""
        z = np.dot(self.weights, inputs) + self.bias
        return self.activation(z)

    def train(self, training_inputs, labels, epochs=100):
        """Train the perceptron using the provided training data."""
        self.errors_ = []
        for epoch in range(epochs):
            errors = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                # Update weights and bias
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
                errors += int(error != 0.0)
            self.errors_.append(errors)
            print(f"Epoch {epoch+1}/{epochs} - Misclassifications: {errors}")
            if errors == 0:
                print(f"Training converged after {epoch+1} epochs.")
                break
        return self

    def accuracy(self, X, y):
        """Calculate the accuracy of the model."""
        predictions = np.array([self.predict(xi) for xi in X])
        return np.mean(predictions == y)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # Flatten and make predictions
    grid_points = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = np.array([classifier.predict(point) for point in grid_points])
    Z = Z.reshape(xx1.shape)
    # Contour plot
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl,
                    edgecolor='black')

def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    # Column names as per the dataset's description
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv(url, header=None, names=column_names)
    
    # Select the first 100 samples (Iris-setosa and Iris-versicolor)
    df = df.iloc[0:100]
    
    # Extract features and labels
    X = df[['sepal_length', 'petal_length']].values
    y = df['class'].values
    
    # Encode labels: Iris-setosa as -1, Iris-versicolor as 1
    y = np.where(y == 'Iris-setosa', -1, 1)
    
    return X, y

def standardize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_std = (X - mean) / std
    return X_std

def main():
    # Load dataset
    X, y = load_data()
    
    # Feature scaling
    X_std = standardize_features(X)
    
    # Initialize Perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1, 
                            weight_init='random', bias_init='zero')
    
    # Train Perceptron
    perceptron.train(X_std, y, epochs=100)
    
    # Plot decision regions
    plt.figure(figsize=(8,6))
    plot_decision_regions(X_std, y, classifier=perceptron)
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.title('Perceptron - Decision Regions')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Plot training errors over epochs
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.title('Perceptron - Training Errors')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Calculate and print accuracy
    accuracy = perceptron.accuracy(X_std, y)
    print(f'Perceptron classification accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()