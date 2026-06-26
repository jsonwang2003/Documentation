> [!INFO]
> Ensemble learning technique that **combines multiple weak classifiers to create a strong classifier**  
> Works iteratively by **training a sequence of base models**, each focusing more on the **misclassified instances** from previous iterations

- **Developed by**: **Yoav Freund** and **Robert Schapire** (1995)
- **Core Principle**: Boosts performance by sequentially training **weak learners** (typically decision stumps) and adjusting weights to emphasize **hard-to-classify instances**
- **Search Strategy**:
	- Assigns **weights to observations**
	- Increases weight for **misclassified samples**
	- Aggregates predictions via **weighted voting**
	- Continues until a set number of learners are trained or perfect classification is achieved

## Workflow

1. **Model Initialization**
	- Start with equal weights for all training samples
2. **Sequential Training**
	- Train weak learners (e.g., decision stumps)
	- Update sample weights based on classification errors
	- Repeat for predefined number of iterations
3. **Prediction & Evaluation**
	- Aggregate predictions from all learners using **weighted votes**
	- Evaluate using metrics:
		- **Accuracy**
		- **Classification Report** (includes **precision**, **recall**, **F1-score**)

## Code Example

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize AdaBoost classifier with decision stumps as base learners
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Train the model
adaboost.fit(X_train, y_train)

# Make predictions
y_pred = adaboost.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Classification Accuracy: {accuracy:.4f}')
print('Classification Report:\n', report)
```
## Advantages

- Improves performance of **weak learners**
	- Highly effective for **classification problems**
- **Resistant to overfitting**
	- Especially when using **decision stumps**
- **Easy to implement** and tune
- Works well with **varied data types**

## Disadvantages

- **Sensitive to noise and outliers**
	- Misclassified samples receive higher weights
	- Can amplify errors and reduce stability
- Not ideal for **extremely large datasets**
	- **Computationally expensive** due to iterative nature
- Less flexible in capturing **complex patterns** compared to gradient boosting