import numpy as np
from sklearn.model_selection import train_test_split

# Linear Regression with Lasso regularization

class LassoRegression:
    def __init__(self, alpha=1.0, iterations=1000, learning_rate=0.01):
        self.alpha = alpha  # Regularization parameter
        self.iterations = iterations  # Number of iterations for gradient descent
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.coefficients_ = None  # Coefficients of the model

    def gradient_descent(self, X, y):
        n_samples = len(y)
        
        for _ in range(self.iterations):
            # Calculate predictions
            y_pred = np.dot(X, self.coefficients_)
            # Calculate residuals
            residuals = y_pred - y
            
            # Calculate gradients
            gradient = (1 / n_samples) * np.dot(X.T, residuals)
            # Apply L1 regularization (Lasso penalty)
            l1_penalty = self.alpha * np.sign(self.coefficients_)
            
            # Update coefficients
            self.coefficients_ -= self.learning_rate * (gradient + l1_penalty)

    def fit(self, X, y):
        # Initialize coefficients
        n_features = X.shape[1]
        self.coefficients_ = np.zeros(n_features)

        # Call the gradient descent method to optimize coefficients
        self.gradient_descent(X, y)

    def predict(self, X):
        return np.dot(X, self.coefficients_)

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

# Example Usage

# Generate synthetic data for demonstration
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples and 5 features
true_coefficients = np.array([3.0, 2.0, 0.0, 0.0, 1.0])  # True coefficients with some zeros for sparsity
y = np.dot(X, true_coefficients) + np.random.randn(100) * 0.5  # Add some noise

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of LassoRegression
lasso_model = LassoRegression(alpha=0.1, iterations=1000, learning_rate=0.01)

# Fit the model to the training data
lasso_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso_model.predict(X_test)

# Calculate Mean Squared Error on test set
mse = lasso_model.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Output the learned coefficients
print(f"Learned Coefficients: {lasso_model.coefficients_}")
