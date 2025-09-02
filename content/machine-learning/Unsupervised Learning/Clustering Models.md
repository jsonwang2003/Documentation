# K-Means Clustering

> [!INFO]
> groups a given dataset into a **pre-defined number of clusters (k) based on feature similarity**
- works iteratively
	1. assign each data point to the nearest cluster center (centroid)
	2. recalculates the centroids until convergence
- Objective is to **minimize the sum of squares distances between data points and their corresponding centroids**
- Algorithm process
	1. initialize k centroids randomly or using methods like k-means++
	2. assign each point to the nearest centroid
	3. update centroids based on mean of assigned points
	4. repeat steps 2 and 3 until centroids stabilize

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Generate synthetic customer data
np.random.seed(42)
data = {
    "Annual_Spending": np.random.randint(1000, 5000, 100),
    "Purchase_Frequency": np.random.randint(1, 20, 100)
}

df = pd.DataFrame(data)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualizing clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Annual_Spending'], y=df['Purchase_Frequency'], hue=df['Cluster'], palette='viridis')
plt.xlabel('Annual Spending')
plt.ylabel('Purchase Frequency')
plt.title('Customer Segmentation using K-Means')
plt.show()

# Display the clustered data
import ace_tools as tools
tools.display_dataframe_to_user(name="K-Means Clustered Data", dataframe=df)
```
### Advantages

- computationally efficient
- easy to implement and interpret
- k-means is highly flexible
### Disadvantages

- sensitive to the choice of k
	- elbow method or silhouette can help determine optimal k
- assumes spherical clusters of equal variance
- outliers significantly distort result
# Hierarchical Clustering

> [!INFO]
> A powerful unsupervised machine learning technique used for grouping data points into a hierarchy of nested clusters

- does not require the pre-specification of the number of clusters
- Algorithm Process: constructs a hierarchy of clusters by adopting either approach
	- bottom-up (agglomerative)
		- more commonly used
		1. begins by treating each data point as an individual cluster
		2. successively merges them based on their similarity
		3. repeat until a single cluster remains
	- top-down (divisive)
		1. starts with single cluster containing all data points
		2. recursively splits them into smaller clusters
- relies on these to **determine the similarity between clusters**
	- distance metric (Euclidean distance)
	- a linkage criterion (single, complete, average)
- final structure represented as a dendrogram
	- helps visualize the merging / splitting process and aids in determining the optimal number of clusters

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Creating a synthetic dataset
np.random.seed(42)
data = pd.DataFrame({
    'Annual_Spending': np.random.randint(500, 5000, 20),  # Simulated spending amounts
    'Purchase_Frequency': np.random.randint(1, 50, 20)  # Simulated frequency of purchases
})

# Standardizing data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Plotting the dendrogram
plt.figure(figsize=(8, 5))
dendrogram = sch.dendrogram(sch.linkage(data_scaled, method='ward'))
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Applying hierarchical clustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
clusters = hc.fit_predict(data_scaled)

# Adding cluster labels to the dataset
data['Cluster'] = clusters
import ace_tools as tools
tools.display_dataframe_to_user(name="Hierarchical Clustering Results", dataframe=data)
```
### Advantages

- interpretability
	- through the dendrogram visualization
- no prior specification
	- flexible in exploratory data analysis
- deterministic
	- result remain consistent unless distance metric / linkage criterion is changed
- well suited for small to moderately sized datasets
	- understanding relationships between data points is as important as clustering itself
### Disadvantages

- computationally expensive
	- time complexity: $O(n^3)$
- sensitive to outliers
	- can significantly distort the clustering datasets by **affecting the linkage calculations**
- cannot be undone when merge / split has occurred

# Density-Based Spatial Clustering of Application with Noise

>[!INFO]
>An unsupervised machine learning algorithm used for clustering data points based on density

- identifies clusters by looking at the density distribution of points in a given dataset
- Algorithm Process
	1. defining 2 parameters
		- epsilon ($\epsilon$):	
			- the neighborhood radius around a point
		- minPts
			- the minimum number of points required to form a dense region
	2. Grouping points based on the 2 parameters
		- Core Point
			- A point with at least "minPts" neighbors within the $\epsilon$-radius
		- Border point
			- A point that is not a core point and falls within the neighborhood of a core point
		- Noise
			- Any point that does not belong to a cluster

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# Generate synthetic transaction data (two interleaving patterns)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Convert results to DataFrame for analysis
df = pd.DataFrame(X_scaled, columns=['Transaction Amount', 'Transaction Frequency'])
df['Cluster'] = labels

# Visualizing the results
plt.figure(figsize=(8,6))
unique_labels = set(labels)
colors = [plt.cm.jet(float(i) / max(unique_labels + {1})) for i in unique_labels]

for label, color in zip(unique_labels, colors):
    subset = df[df['Cluster'] == label]
    plt.scatter(subset['Transaction Amount'], subset['Transaction Frequency'], color=color, label=f'Cluster {label}', edgecolors='k')

plt.xlabel("Transaction Amount (Standardized)")
plt.ylabel("Transaction Frequency (Standardized)")
plt.title("DBSCAN Clustering of Financial Transactions")
plt.legend()
plt.show()
```
### Advantages

- Able to detect clusters of arbitrary shapes without requiring the user to specify the number of clusters
- Robust to noise and outliers
- Algorithm is density-based
	- Can uncover meaningful clusters even when traditional methods fail
### Disadvantages

- Performance is sensitive to the selection of $\epsilon$ and minPts
- Struggles with high-dimensional data
- Not suitable for large datasets
	- Require a nearest-neighbor search for each point
	- Computationally expensive

# Gaussian Mixture Models

>[!INFO]
>A probabilistic clustering technique that models data as a mixture of multiple Gaussian distributions

- Assigns probabilities to each point belonging to different clusters
	- Allow soft clustering
	- Useful when **clusters exhibit ellipsoidal** or **overlapping shapes**
- Algorithm Process: 
	- [[Expectation-Maximization algorithm (EM)]]
	- Define parameters like these to each Gaussian component
		- Mean
		- Covariance
		- weight
	- Iteratively refines the parameters by alternating between
		- Expectation step: assigning probabilities to data points
		- Maximization step: updating the parameters based on these probabilities

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic data
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.6)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```
### Advantages

- Able to model elliptical and overlapping clusters
- Soft clustering approach
	- Provide valuable insights into the degree of belongingness of each data point to different clusters
- Can handle varying cluster densities
	- By adjusting the covariance matrices of each Gaussian component
### Disadvantages

- Sensitive to initialization of parameters
	- Can lead to convergence at local optima
	- Mitigate by multiple initializations
- Assumes underlying data distribution follows mixture of Gaussians
- Computationally expensive
	- Iterative nature of EM algorithm

# Mean-Shift Clustering

>[!INFO]
>A **non-parametric**, **iterative clustering algorithm** that does not require the ***pre-specification of the number of clusters***. 
>Identifies **dense areas in the feature space** by **shifting data points toward regions of higher density**

- Algorithm Process
	- Iteratively updating each point's position based on a kernel density estimate (Gaussian Kernel) until convergence
		- Points are shifted to stable position
- Core idea
	- Move each point toward the mean of it's neighborhood
	- Gradually forming clusters around high-density areas
- Adaptively discover the optimal number of clusters based on the data distribution
- The radius defining a neighborhood around each point (bandwidth) has significant impact on clustering performance

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# Generate synthetic customer data (spending, frequency)
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Estimate the optimal bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)

# Apply Mean-Shift clustering
ms = MeanShift(bandwidth=bandwidth)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# Convert to DataFrame for analysis
df = pd.DataFrame(X, columns=["Spending", "Frequency"])
df["Cluster"] = labels

# Display cluster statistics
import ace_tools as tools
tools.display_dataframe_to_user(name="Mean-Shift Clustering Results", dataframe=df)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k', alpha=0.7)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Mean-Shift Clustering of Customer Segments')
plt.xlabel('Spending')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```
### Advantages

- Able to discover the optimal number of clusters without requiring prior knowledge
- Well suited for detecting arbitrary shaped clusters
- Does not assume clusters must be spherical
- Avoids the issue of local minima with no reliance on initialization points
### Disadvantages

- Computational complexity is high
- Sensitive to bandwidth selection
- May struggle with high-dimensional data

# Affinity Propagation

>[!INFO]
>Clustering algorithm that identifies exemplars (representative data points) through a message-passing mechanism

- Determines the optimal number of clusters based on data similarities
- Algorithm Process
	- Iteratively refining the assignment of data points through the exchange of 2 messages
		1. Responsibility: measures how well a data point should serve as an exemplar for another
		2. Availability: quantifies the degree to which a data point is suited to be an exemplar
	- Continues updating until convergence is reached

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
### Advantages

- Does not require number of clusters to be specified beforehand
- Effectively identifies representative exemplars within clusters
- Performs well with non-linearly separable data
### Disadvantages

- High computational complexity
- Requires careful tuning of preference parameter
- Sensitive to noise and outliers

# Spectral Clustering

> [!INFO]
> - Powerful unsupervised learning technique that uses graph theory and eigenvalues of matrices to partition data into clusters
- Leverages the eigenvalues of the similarity matrix to perform dimensionality reduction before applying clustering algorithms
	- Useful for datasets with complex structures, non-convex shapes, or data points that do not conform well to Euclidean distance-based clustering
- Algorithm Process
	1. construct an affinity representing the similarities between points
	2. Compute the graph Laplacian
	3. Use eigenvector decomposition to embed the data in a lower-dimensional space where clustering can be applied effectively

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

# Generate synthetic dataset with non-linearly separable clusters
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# Apply Spectral Clustering
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
labels = spectral.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.title("Spectral Clustering on Non-Convex Data")
plt.show()
```
### Advantages

- Able to handle complex and non-convex cluster structures
- Useful when working with graph-based data
- Does not require assumptions about cluster shape
### Disadvantages

- Computationally expensive
	- $O(n^3)$
- Requires careful tuning of hyperparameters
	- Affinity matrix construction method and number of clusters
- Rely on the similarity graph