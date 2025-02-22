from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# This is intended to compare different model perform in doing the digit recognition
# KNN vs Logistics Regression vs SVC

digits = load_digits()

xs = digits.data
ys = digits.target

def get_accuracy(ys, ys_pred):
    if len(ys) != len(ys_pred):
        raise ValueError("ys and ys_pred must have the same length")
    if len(ys) == 0:
        return 0.0
    correct_count = sum(1 for y_true, y_pred in zip(ys, ys_pred) if y_true == y_pred)
    acc = correct_count / len(ys)
    return acc

x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.3, random_state=42)

C_list = [0.001, 0.01, 0.1, 1, 10, 100]
solver_list = ['lbfgs', 'liblinear']
max_iter = 1000

result1 = {}

print("\nLogistic Regression performance for various hyperparameter combinations:")
for solver in solver_list:
    for C in C_list:
        # Create and fit the logistic regression model with the given hyperparameters.
        clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = get_accuracy(y_test, y_pred)
        result1[(solver, C)] = accuracy
        print(f"Solver: {solver:9s}, C: {C:6.3f} -> Test Accuracy: {accuracy:.4f}")



best_params = max(result1, key=result1.get)
best_accuracy = result1[best_params]
print(f"\nBest hyperparameters: Solver = {best_params[0]}, C = {best_params[1]}, Accuracy = {best_accuracy:.4f}")

result2 = {}

C_list = [0.1, 1, 10, 100, 1000]
gamma_list = [0.001, 0.01, 0.1, 1]
kernel = 'rbf'

for C in C_list:
    for gamma in gamma_list:
        # Create and fit the SVC model with the specified hyperparameters
        svc = SVC(C=C, gamma=gamma, kernel=kernel)
        svc.fit(x_train, y_train)
        
        # Predict on the test set and evaluate accuracy
        y_pred = svc.predict(x_test)
        accuracy = get_accuracy(y_test, y_pred)
        result2[(C, gamma)] = accuracy
        print(f"C: {C:7.3f}, gamma: {gamma:7.3f} -> Test Accuracy: {accuracy:.4f}")

# Identify the best hyperparameter combination
best_params = max(result2, key=result2.get)
best_accuracy = result2[best_params]
print(f"\nBest hyperparameters: C = {best_params[0]}, gamma = {best_params[1]}, with Test Accuracy = {best_accuracy:.4f}")

results_knn = {}
n_neighbors_list = [3, 5, 7, 9,]
weights_list = ['uniform', 'distance']
p_list = [1, 2]  # p = 1: Manhattan distance, p = 2: Euclidean distance

print("\nKNeighborsClassifier performance for various hyperparameter combinations:")
# Loop over all combinations of hyperparameters
for n_neighbors in n_neighbors_list:
    for weight in weights_list:
        for p in p_list:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight, p=p)
            knn.fit(x_train, y_train)
            y_pred_knn = knn.predict(x_test)
            accuracy_knn = get_accuracy(y_test, y_pred_knn)
            
            key = (n_neighbors, weight, p)
            results_knn[key] = accuracy_knn
            print(f"n_neighbors: {n_neighbors:2}, weights: {weight:8}, p: {p} -> Test Accuracy: {accuracy_knn:.4f}")

# Identify the best hyperparameter combination
best_params = max(results_knn, key=results_knn.get)
best_accuracy = results_knn[best_params]

print("\nBest hyperparameter combination for KNN:")
print(f"n_neighbors: {best_params[0]}")
print(f"weights: {best_params[1]}")
print(f"p: {best_params[2]}")
print(f"Achieved Test Accuracy: {best_accuracy:.4f}")