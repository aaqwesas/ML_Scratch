import numpy as np
from sklearn.model_selection import train_test_split

# Linear Regression with Ridge regularization

class RidgeRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01, max_iterations=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iterations):
            # Compute predictions
            y_pred = X @ self.weights + self.bias

            # Compute gradients
            dw = (1 / n_samples) * (X.T @ (y_pred - y)) + (self.alpha / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return X @ self.weights + self.bias

# Generate synthetic data
X = np.random.rand(100, 3)
true_weights = np.array([3.0, -2.0, 1.0])
y = X @ true_weights + np.random.randn(100) * 0.5

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Ridge Regression model using gradient descent
ridge = RidgeRegression(alpha=1.0, learning_rate=0.01, max_iterations=1000)
ridge.fit(X_train, y_train)

# Make predictions
y_pred = ridge.predict(X_test)

# Output the learned weights and bias
print("Learned weights:", ridge.weights)
print("Learned bias:", ridge.bias)
