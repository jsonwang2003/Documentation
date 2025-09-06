> [!INFO]
> Simple but effective **non-parametric machine learning technique** used for both **classification** and **regression** tasks  
> Operates on the principle of **similarity**, assigning observations based on the **majority vote or average of their k-nearest neighbors**

- **Developed by**: **Fix and Hodges** (1951)
- **Core Principle**: Stores the entire training dataset and makes predictions by measuring **distance** to nearby points
- **Search Strategy**:
	- No explicit model is built
	- Uses **Euclidean**, **Manhattan**, or **Minkowski distance** to find nearest neighbors
	- Choice of **k** affects bias-variance tradeoff
		- Small k → sensitive to local patterns
		- Large k → smoother decision boundaries, risk of misclassifying minority cases

## Workflow

1. **Data Preparation**
	- Standardize features to ensure fair distance comparisons
2. **Model Training**
	- Store training data
	- No fitting required beyond data storage
3. **Prediction & Evaluation**
	- Predict class based on **majority vote** among k-nearest neighbors
	- Evaluate using metrics:
		- **Accuracy**
		- **Classification Report** (includes **precision**, **recall**, **F1-score**)

## Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Sample dataset: Customer age, income, and spending category (0 = Low, 1 = High)
data = {
    'Age': [25, 34, 45, 52, 23, 40, 60, 48, 33, 29],
    'Income': [40000, 60000, 80000, 75000, 30000, 72000, 95000, 88000, 54000, 50000],
    'Spending_Category': [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]  # Target variable
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Splitting features and target
X = df[['Age', 'Income']]
y = df['Spending_Category']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying k-NN classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Predictions
y_pred = knn.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)
```
## Advantages

- **Simple** and intuitive
- Effective with **smaller datasets**
- **Highly interpretable**
	- No training phase
	- Naturally adapts to **non-linear decision boundaries**

## Disadvantages

- **Computationally expensive** for large datasets
	- Requires distance calculation to all training points
- Suffers from **curse of dimensionality**
	- Performance degrades with many features
- **Sensitive to noise and imbalanced data**
	- Outliers or skewed class distributions can distort predictions
	- Mitigation strategies:
		- Optimize **k**
		- Use **feature selection**
		- Apply **distance weighting**