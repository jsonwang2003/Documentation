> [!INFO]
> Powerful supervised learning algorithm used primarily for **classification tasks**, but also applicable to **regression problems**

- **Developed by**: **Vladimir Vapnik** and **Alexey Chervonenkis** (1963; popularized in the 1990s)
- **Core Principle**: Identifies the **optimal hyperplane** that best separates data points into distinct classes by **maximizing the margin** between them
- **Search Strategy**:
	- If data is **linearly separable**, find the hyperplane with **maximum margin**
	- If not, apply the **kernel trick** to transform data into a higher-dimensional space
	- Common kernels:
		- **Linear**
		- **Polynomial**
		- **Radial Basis Function (RBF)**
		- **Sigmoid**
	- Robust to **high-dimensional data** and grounded in **statistical learning theory**

## Workflow

1. **Data Preparation**
	- Standardize or normalize features
	- Choose appropriate kernel and hyperparameters
2. **Model Training**
	- Fit SVM to training data using selected kernel
	- Optimize margin and support vectors
3. **Prediction & Evaluation**
	- Predict class labels on test data
	- Evaluate using metrics:
		- **Confusion Matrix**: shows **true positives**, **true negatives**, **false positives**, **false negatives**
		- **Classification Report**: includes **precision**, **recall**, **F1-score**

## Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Generating synthetic data
np.random.seed(42)
num_samples = 500

# Creating two classes: Legitimate (0) and Fraudulent (1)
X_legit = np.random.normal(loc=50, scale=10, size=(num_samples // 2, 2))
X_fraud = np.random.normal(loc=70, scale=10, size=(num_samples // 2, 2))
X = np.vstack((X_legit, X_fraud))
y = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training SVM with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Plot decision boundary
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k', alpha=0.7)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Classification of Fraudulent Transactions")
plt.show()

# Display results
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
```
## Advantages

- Effective in **high-dimensional spaces**
	- Suitable for **text classification**, **image recognition**, **fraud detection**
- Excels with **small to medium-sized datasets** and **complex decision boundaries**
	- Leverages the **kernel trick**
- **Less prone to overfitting** compared to some models

## Disadvantages

- **Computationally intensive** for large datasets
	- Training complexity scales poorly with sample size
- Requires **careful hyperparameter tuning**
	- Kernel type, regularization parameter (**C**), and **gamma** for RBF
- Does not provide **probabilistic outputs** directly
	- Limitation when **confidence scores** are neede