> [!INFO]
> Powerful supervised learning technique used in machine learning for **predictive modeling**  
> Works by **recursively partitioning the dataset into subsets based on feature values**, creating a **tree-like structure**

- **Developed by**: **Breiman et al.** (1986, CART formalization)
- **Core Principle**: Splits data into branches based on feature thresholds to form interpretable decision rules
- **Search Strategy**:
	- Uses **CART (Classification and Regression Trees)** algorithm
	- Measures **Gini impurity** or **entropy** to determine best splits
	- Builds tree until stopping criteria are met (e.g., max depth, minimum samples)
## Workflow

1. **Tree Construction**
	- Select best feature and threshold to split data
	- Recursively partition data into branches
	- Assign leaf nodes with predicted class labels
2. **Evaluation**
	- Assess model using classification metrics
	- Evaluate using metrics:
		- **Accuracy**: proportion of correct predictions
		- **Classification Report**: includes **precision**, **recall**, **F1-score**

## Code Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset: Customer churn prediction
data = {'MonthlyBill': [50, 20, 80, 30, 90, 40, 70, 60, 25, 85],
        'ContractType': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],  # 1: Long-term, 0: Short-term
        'CallDrops': [2, 5, 1, 6, 0, 7, 1, 3, 5, 1],
        'Churn': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]  # 1: Churn, 0: No churn
}

# Create DataFrame
df = pd.DataFrame(data)

# Split data into features and target variable
X = df[['MonthlyBill', 'ContractType', 'CallDrops']]
y = df['Churn']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Plot the decision tree
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plot_tree(clf, feature_names=['MonthlyBill', 'ContractType', 'CallDrops'], 
          class_names=['No Churn', 'Churn'], filled=True)
plt.show()
```
## Advantages

- **Highly interpretable**
	- Excellent for explain ability and stakeholder communication
- Handles both **numerical and categorical data**
- Requires **minimal preprocessing**
- Captures **complex relationships** without explicit feature transformations
## Disadvantages

- **Prone to overfitting**
	- Especially with deep trees
	- Mitigation techniques:
		- **Pruning**
		- **Max depth constraints**
		- **Ensemble methods** like **Random Forests**
- **Sensitive to small data variations**
	- Small changes can lead to different tree structures
- **Bias toward dominant classes** in imbalanced datasets
	- May require **resampling** or **class weighting**