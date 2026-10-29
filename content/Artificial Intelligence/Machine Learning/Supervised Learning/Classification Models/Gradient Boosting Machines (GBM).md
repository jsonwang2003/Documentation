> [!INFO]
> Powerful **ensemble learning technique** designed for **predictive modeling tasks**, particularly **classification** and **regression**

- **Developed by**: **Friedman (2001)**; **XGBoost** implementation by **Tianqi Chen** (2016)
- **Core Principle**: Sequentially trains **weak learners** to correct errors from previous iterations using **gradient descent**
- **Search Strategy**:
	- Minimizes a **differentiable loss function**
	- Each tree is trained on the **negative gradient** (residuals) of the loss
	- Aggregates predictions in an **additive manner**
	- Refines decision boundaries iteratively to improve classification accuracy

## Workflow

1. **Data Preparation**
	- Extract features and target variable
	- Split into training and testing sets
2. **Model Training**
	- Fit gradient boosting model using training data
	- Optimize using **learning rate**, **tree depth**, and **number of estimators**
3. **Prediction & Evaluation**
	- Predict class labels on test data
	- Evaluate using metrics:
		- **Accuracy**
		- **Classification Report** (includes **precision**, **recall**, **F1-score**)

## Code Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Generate synthetic dataset (for demonstration)
np.random.seed(42)
data_size = 1000
df = pd.DataFrame({
    'MonthlyCharges': np.random.uniform(20, 100, data_size),
    'Tenure': np.random.randint(1, 72, data_size),
    'SupportCalls': np.random.randint(0, 10, data_size),
    'TotalUsage': np.random.uniform(100, 5000, data_size),
    'Churn': np.random.choice([0, 1], size=data_size, p=[0.8, 0.2])
})

# Splitting the dataset
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train GBM model
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
```
## Advantages

- Achieves **high predictive accuracy** via **gradient-based optimization**
- Handles **complex, non-linear relationships** between features
- Can manage **missing data** effectively
	- Reduces need for extensive preprocessing

## Disadvantages

- **Computationally expensive**
	- Sequential learning slows training on large datasets
- **Prone to overfitting** if not carefully tuned
	- Requires hyperparameter tuning:
		- **Learning rate**
		- **Tree depth**
		- **Number of estimators**
- **Interpretability** is limited
	- Decision process is less intuitive than simpler models