> [!INFO]
> Widely used statistical method for **binary classification problems**, where the outcome variable is categorical and typically represents two classes

- **Developed by**: **David Cox** (1958)
- **Core Principle**: Estimates the **probability that a given input belongs to a particular class** using the **logistic (sigmoid) function**
- **Search Strategy**:
	- Trained using **maximum likelihood estimation**
	- Adjusts coefficients to minimize the difference between predicted and actual values
	- Outputs probabilities in the range **0 to 1** for decision-making

## Workflow

1. **Data Preparation**
	- Select relevant features and standardize inputs
2. **Model Training**
	- Fit logistic regression using training data
	- Optimize coefficients via **maximum likelihood**
3. **Prediction & Evaluation**
	- Predict class labels on test data
	- Evaluate using metrics:
		- **Accuracy**: proportion of correct predictions
		- **Confusion Matrix**: shows **true positives**, **true negatives**, **false positives**, **false negatives**
		- **Classification Report**: includes **precision**, **recall**, **F1-score**

## Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler

# Generate synthetic customer purchase data
np.random.seed(42)
data = np.array([
    [500, 20, 3], [550, 18, 2], [600, 22, 5],  # High spenders
    [100, 5, 0], [120, 4, 1], [130, 6, 0],     # Low spenders
    [300, 10, 2], [310, 12, 3], [320, 9, 1],   # Mid-range spenders
    [400, 15, 4], [420, 14, 3], [430, 16, 4]   # Upper-mid-range spenders
])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply Affinity Propagation
aff_prop = AffinityPropagation(random_state=42)
clusters = aff_prop.fit_predict(data_scaled)

# Convert results to a DataFrame
customer_df = pd.DataFrame(data, columns=['Annual Spend ($)', 'Purchases per Month', 'Returns'])
customer_df['Cluster'] = clusters

# Display the clustered data
import ace_tools as tools
tools.display_dataframe_to_user(name="Affinity Propagation Clustering Results", dataframe=customer_df)

# Plot clusters
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', edgecolors='k', s=100)
plt.xlabel('Annual Spend ($)')
plt.ylabel('Purchases per Month')
plt.title('Customer Clustering using Affinity Propagation')
plt.show()
```
## Advantages

- **Interpretability**
	- Coefficients indicate how each independent variable influences the likelihood of an outcome
	- Performs well when the relationship is **linear in the log-odds space**
- **Computationally efficient**
	- Requires less training time than more complex models
## Disadvantages

- Assumes **linearity** between independent and dependent variables
	- May not always hold true
- Struggles with **high-dimensional data**
	- Unless combined with **feature selection** or **regularization techniques** like **L1 (Lasso)** and **L2 (Ridge)**
- Less effective for **complex, non-linear relationships**
	- More sophisticated models may yield better results
  