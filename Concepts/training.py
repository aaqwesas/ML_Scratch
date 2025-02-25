from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load example dataset
data = load_iris()
X = data.data
y = data.target

# Always spilit the data before standardizing the data, then the model see the testing data as well which is data leakage,avoid leading to misleading model performance

# Perform a stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# Standardize the features using StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# Predict on the test set
ppn = Perceptron(eta0=0.01,random_state=1)
ppn.fit(X_train_std,y_train)
y_pred = ppn.predict(X_test_std)

#print out the accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Accuracy:", ppn.score(X_test_std,y_test))

