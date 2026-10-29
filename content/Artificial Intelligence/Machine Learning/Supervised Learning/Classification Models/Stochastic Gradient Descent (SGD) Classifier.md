> [!INFO]
> Optimization algorithm used in machine learning for **efficiently training models**, particularly those involving **large-scale datasets**

- **Developed by**: **Herbert Robbins** and **Sutton Monro** (1951, foundational SGD theory)
- **Core Principle**: Updates model parameters using **one sample at a time**, making it **computationally efficient** for large datasets
- **Search Strategy**:
	- Applies **stochastic gradient descent optimization**
	- Supports multiple **loss functions**:
		- **Hinge loss** (for Support Vector Machines)
		- **Log loss** (for Logistic Regression)
	- Often used with **linear models** for classification tasks
	- Enables **online learning** by updating weights incrementally

## Workflow

1. **Data Preparation**
	- Standardize features to ensure consistent gradient updates
2. **Model Training**
	- Fit SGD classifier using chosen loss function
	- Update weights per sample using gradient of loss
3. **Prediction & Evaluation**
	- Predict class labels on test data
	- Evaluate using metrics:
		- **Accuracy**
		- **Classification Report** (includes **precision**, **recall**, **F1-score**)

## Code Example

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generating a synthetic dataset (simulating e-commerce browsing behavior)
np.random.seed(42)
n_samples = 1000

# Features: products viewed, time spent (minutes), cart added (binary)
X = np.random.rand(n_samples, 3) * [10, 50, 1]  # Scaling different ranges
y = np.random.choice([0, 1], size=n_samples)  # Binary target: 1 (Purchase), 0 (No Purchase)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the SGD Classifier
sgd_clf = SGDClassifier(loss='log', max_iter=1000, learning_rate='optimal', random_state=42)
sgd_clf.fit(X_train, y_train)

# Predictions
y_pred = sgd_clf.predict(X_test)

# Evaluating the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
```
## Advantages

- **Scalable** for large datasets
	- Updates weights incrementally
	- Suitable for **real-time applications**
- **Fast and memory-efficient**
	- Especially effective with **high-dimensional data**
- Supports **online learning**
	- Model can be updated continuously with new data

## Disadvantages

- **High variance and instability**
	- Single-sample updates introduce **stochastic noise**
	- May converge to **suboptimal solutions**
- **Sensitive to hyperparameters**
	- Learning rate and number of iterations must be tuned carefully
- Not ideal for **small datasets**
	- Random updates may lead to **poor generalization**
	- May require **batch or mini-batch gradient descent** for stability