> [!INFO]
> Probabilistic classification algorithm based on **Bayes' Theorem**  
> Calculates the **posterior probability** of a class given a set of features by multiplying the **prior probability** of the class with the **likelihood** of the features occurring within that class, normalized by the **probability of the features across all classes**


$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

- **Developed by**: **Thomas Bayes** (18th century); formalized in modern machine learning contexts in the 20th century
- **Core Principle**: Assumes **conditional independence** between features given the class label
- **Search Strategy**:
	- Estimate **prior probabilities** for each class
	- Compute **likelihoods** for each feature given the class
	- Apply **Bayes’ Theorem** to calculate posterior probabilities
	- Select class with highest posterior

## Workflow

1. **Data Preparation**
	- Extract features and target variable
	- Split into training and testing sets
2. **Model Training**
	- Fit Naïve Bayes model (e.g., GaussianNB for continuous features)
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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'Salary': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000],
    'Purchased': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # 0 = No, 1 = Yes
}
df = pd.DataFrame(data)

# Split dataset into features and target variable
X = df[['Age', 'Salary']]
y = df['Purchased']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naïve Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
```
## Advantages

- **Computationally efficient**
	- No iterative parameter tuning required
- Performs well with **categorical** and **text data**
	- Ideal for **spam detection**, **sentiment analysis**, and **recommendation systems**
- Handles **missing values** gracefully
	- Probabilities estimated independently for each feature

## Disadvantages

- Assumes **feature independence**, which is often unrealistic
	- Can lead to **suboptimal performance**
- Struggles with **small datasets**
	- Probability estimates may be unreliable
- Not suitable for **complex relationships** between features