> [!INFO]
> Transforms high-dimensional data into a lower-dimensional space while **preserving as much variance as possible**

- **Developed by**: **Karl Pearson** (1901)
- **Core Principle**: Projects data onto orthogonal components that capture the greatest variance
- **Search Strategy**:
	- Compute **principal components** as linear combinations of original variables
	- Select top components that retain the most **informative variance**
	- Reduce dimensionality while preserving structure

## Workflow

1. **Component Computation**
	- Standardize the dataset
	- Compute **covariance matrix**
	- Perform **eigen decomposition** to extract principal components
2. **Dimensionality Reduction**
	- Rank components by **explained variance**
	- Select top components for projection
	- Transform data into reduced space

## Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Simulated business dataset: sales figures across different product categories
np.random.seed(42)
data = np.random.rand(100, 5) * 1000  # 100 customers, 5 product categories
columns = ['Electronics', 'Clothing', 'Groceries', 'Furniture', 'Sports']

df = pd.DataFrame(data, columns=columns)

# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Applying PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
principal_components = pca.fit_transform(scaled_data)

# Creating a new DataFrame with principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Display results
import ace_tools as tools
tools.display_dataframe_to_user(name="PCA Results", dataframe=pca_df)

# Plot the explained variance
plt.figure(figsize=(6, 4))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Explained Variance by Principal Components')
plt.show()
```

## Advantages

- Efficient for **high-dimensional data**
	- Reduces **noise** and eliminates **redundant features**
- Widely used in **exploratory data analysis**, **pattern recognition**, and **data compression**
- Improves model performance
	- Mitigates **multicollinearity** and enhances **interpretability**

## Disadvantages

- Relies on **linear transformations**
	- May not perform well with **non-linear relationships**
- Principal components are **hard to interpret**
	- Represent **linear combinations** of original features
- Assumes **variance = importance**
	- _**Not always true**_
- Requires **data standardization** for effective performance
- Selecting the **optimal number of components** can be subjective