# Principal Component Analysis

> [!INFO]
> Transforms high-dimensional data into a lower-dimensional space while **preserving as much variance as possible**
- Widely used dimensionality reduction technique in data science and machine learning
- Computes the principal components
	- Uncorrelated variables formed as linear combinations of the original variables
- First principal component captures the highest variance in the dataset
- Each subsequent component capturing the maximum possible variance orthogonal to the previous components
- By selecting only the most significant principal components, while retaining essential information, helps in
	- Reducing computational complexity
	- Mitigating multilinearity
	- Improving model performance

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
### Advantages

- Able to handle high-dimensional data efficiently
	- Reduces noise and eliminating redundant features
- Used in exploratory data analysis, pattern recognition, and data compression
- Improves model performance
	- Mitigating multicollinearity and enhancing interpretability
### Disadvantages

- Relies on linear transformations &rarr; may not perform well when dealing with non-linear relationships in data
- Principle components often difficult to interpret
	- They are linear combinations of the original variables rather than directly meaningful features
- Assumes variance = importance
	- ***Not always true***
- Requires data standardization for effective performance
- Subjective opinion when selecting optimal number of principal components

# t-Distributed Stochastic Neighbor Embedding (t-SNE)

> [!INFO]
> Non-linear dimensionality reduction technique that is widely used for visualizing high-dimensional data in a low-dimensional space (typically 2 or 3 dimensions)

- Developed by Laurens Van Der Maaten and Geoffrey Hinton
- Converts high-dimensional similarities into probabilities and optimally positions data points in lower dimensions while preserving their local relationships
- Focuses on maintaining the local structure of data by minimizing the [[Kullback-Leibler divergence (KL divergence)]] between joint probability distributions in the original and reduced spaces
	- Achieved through an iterative gradient descent process
- Useful in clustering applications
	- Reveals complex patterns and relationships that may not be easily discernible in raw data

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
### Advantages

- Able to capture and preserve **local relationships in data**
- Adapts to non-linear structures and provides intuitive visual groupings
### Disadvantages

- Computationally expensive
- Results can vary with different parameter settings
	- Perplexity
	- Learning rate
- Does not provide explicit cluster assignments
- May distort global relationships
	- Focuses on local structure

# Uniform Manifold Approximation and Projection (UMAP)

> [!INFO]
> State-of-the-art dimensionality reduction technique for **visualizing high-dimensional data while preserving its structure**

- Non-linear technique built upon principles of topological data analysis and manifold learning
	- Effective for capturing **both global and local structures within datasets**
- Constructs high-dimensional graph representation of the data and then optimizes a low-dimensional graph to approximate the original structure

```python
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate synthetic customer data
np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=10, n_classes=4, n_informative=5, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)

# Convert results to DataFrame
df_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
df_umap['Cluster'] = y

# Plot the UMAP results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_umap['UMAP1'], df_umap['UMAP2'], c=df_umap['Cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Cluster")
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Projection of Customer Data')
plt.show()
```
### Advantages

- Able to preserve both local and global structures within high-dimensional data
- Computationally efficient
- Can serve as a pre-processing step for clustering algorithm
- Supports supervised and unsupervised learning
### Disadvantages

- Highly dependent on hyperparameters
	- Number of neighbors
	- Minimum distance
- Does not provide a direct interpretation of feature importance or variance explained

# Independent Component Analysis

> [!INFO]
> Computational technique used to separate a multivariate signal into additive, statistically independent components

- Primarily employed for blind source separation
	- Mixed signals are decomposed into their original, independent sources without prior knowledge about the mixing process
- Seeks to maximize the statistical independence of extracted components
	- Assumes the observed signals are linear mixtures of non-Gaussian source signals
	- Then applies an optimization process to minimize mutual information or maximize non-Gaussian

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Generate synthetic independent signals
np.random.seed(42)
time = np.linspace(0, 8, 1000)
s1 = np.sin(2 * time)  # Sinusoidal source
s2 = np.sign(np.sin(3 * time))  # Square wave source
s3 = np.random.normal(size=1000)  # Gaussian noise source

# Stack sources and mix them linearly
S = np.c_[s1, s2, s3]
A = np.array([[0.5, 1, 0.2], [1, 0.5, 0.3], [0.2, 0.3, 1]])  # Mixing matrix
X = S @ A.T  # Mixed signals

# Apply ICA
ica = FastICA(n_components=3)
S_estimated = ica.fit_transform(X)  # Extract independent components

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(10, 6))
axes[0].plot(S, label=['Source 1', 'Source 2', 'Source 3'])
axes[0].set_title("Original Independent Signals")
axes[0].legend()

axes[1].plot(X, label=['Mixed 1', 'Mixed 2', 'Mixed 3'])
axes[1].set_title("Observed Mixed Signals")
axes[1].legend()

axes[2].plot(S_estimated, label=['ICA Component 1', 'ICA Component 2', 'ICA Component 3'])
axes[2].set_title("Recovered Independent Components using ICA")
axes[2].legend()

plt.tight_layout()
plt.show()
```
### Advantages

- Excels in blind source separation tasks where little is known about the original signals
- Able to extract statistically independent components
- Effective in scenarios where the signals exhibit non-Gaussian distributions
### Disadvantages

- Sensitive to noise and outliers
	- Struggles to accurately separate independent components
- Assumes the number of sources is less than or equal to the number of observed signals
- Reliance on strong assumptions about the independence and non-Gaussian
	- Unreliable components extracted
- Computationally expensive