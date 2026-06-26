> [!INFO]
> Non-linear dimensionality reduction technique that is widely used for **visualizing high-dimensional data** in a low-dimensional space (typically 2 or 3 dimensions)

- **Developed by**: **Laurens van der Maaten** and **Geoffrey Hinton** (2008)
- **Core Principle**: Converts high-dimensional similarities into **probabilities** and positions data points in lower dimensions while preserving **local relationships**
- **Search Strategy**:

	- Minimize **Kullback-Leibler divergence** between joint probability distributions in original and reduced spaces
	- Achieved through **iterative gradient descent**
	- Focuses on **local structure preservation** rather than global geometry

## Workflow

1. **Similarity Computation**
	- Convert pairwise distances into **conditional probabilities**
2. **Embedding Optimization**
	- Initialize low-dimensional representation
	- Minimize **KL divergence** via gradient descent
	- Evaluate using metrics: _Not applicable_

## Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import ace_tools as tools

# Load dataset (Digits dataset for visualization)
digits = load_digits()
X = digits.data
y = digits.target

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_scaled)

# Convert to DataFrame for visualization
df_tsne = pd.DataFrame({'Component 1': X_embedded[:, 0], 
                         'Component 2': X_embedded[:, 1], 
                         'Label': y})

# Display the DataFrame
tools.display_dataframe_to_user(name="t-SNE Results", dataframe=df_tsne)

# Visualization of t-SNE output
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='jet', alpha=0.7)
plt.colorbar(scatter, label="Digit Label")
plt.title("t-SNE Visualization of Digits Dataset")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
```

## Advantages

- Captures and preserves **local relationships** in data
- Adapts to **non-linear structures**
- Provides **intuitive visual groupings**

## Disadvantages

- **Computationally expensive**
- Sensitive to **parameter settings**
	- Perplexity
	- Learning rate
- Does not provide **explicit cluster assignments**
- May distort **global relationships**