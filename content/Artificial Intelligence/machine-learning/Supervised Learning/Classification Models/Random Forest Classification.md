> [!INFO]
> Ensemble learning technique used for **classification** and **regression tasks**  
> Constructs multiple decision trees during training and outputs the **mode of individual tree predictions** for classification (majority vote)

- **Developed by**: **Leo Breiman** and **Adele Cutler** (2001)
- **Core Principle**: Combines predictions from multiple decision trees to improve **generalization** and reduce **overfitting**
- **Search Strategy**:
	- Introduces **randomness** by selecting **random subsets of features and samples** for each tree
	- Ensures **diversity in decision boundaries**
	- Aggregates predictions via **majority voting** (classification) or **averaging** (regression)

## Workflow

1. **Model Construction**
	- Generate multiple decision trees using bootstrapped samples
	- Randomly select features for each split
2. **Prediction & Evaluation**
	- Aggregate predictions from all trees
	- Evaluate using metrics:
		- **Accuracy**
		- **Confusion Matrix**
		- **Classification Report** (includes **precision**, **recall**, **F1-score**)
	- Analyze **feature importance** to interpret model behavior

## Code Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic dataset
np.random.seed(42)
data = pd.DataFrame({
    'tenure': np.random.randint(1, 60, 500),
    'monthly_charges': np.random.uniform(20, 100, 500),
    'total_charges': np.random.uniform(100, 6000, 500),
    'customer_support_calls': np.random.randint(0, 10, 500),
    'contract_type': np.random.choice([0, 1], size=500),  # 0: Month-to-Month, 1: Long-Term
    'churn': np.random.choice([0, 1], size=500)  # 0: No churn, 1: Churn
})

# Split data into training and testing sets
X = data.drop(columns=['churn'])
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Feature importance
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)

# Display results
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

# Plot feature importance
plt.figure(figsize=(8,5))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()
```
## Advantages

- Handles **large datasets** with **high-dimensional features**
- **Less prone to overfitting** than single decision trees
- Supports both **categorical and numerical data**
- Offers **interpretability** through feature importance analysis

## Disadvantages

- **Computationally intensive**
	- Requires significant processing power and memory
	- Slower for **real-time prediction**
- May struggle with **imbalanced datasets**
	- Requires techniques like **class weighting** or **resampling** to mitigate bias