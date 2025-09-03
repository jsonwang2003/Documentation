> [!INFO]
> Computational technique used to separate a **multivariate signal** into additive, **statistically independent components**

- **Developed by**: **Pierre Comon** (1994, formalized ICA as a statistical method)
- **Core Principle**: Decomposes observed signals into statistically independent sources using assumptions of **non-Gaussianity** and **linearity**
- **Search Strategy**:
	- Assumes observed signals are **linear mixtures** of **non-Gaussian sources**
	- Applies optimization to **maximize independence**
- Minimizes **mutual information**
- Maximizes **non-Gaussianity** (e.g., via kurtosis or negentropy)

## Workflow

1. **Signal Preparation**
	- Collect multivariate observations
	- Center and whiten the data
2. **Component Extraction**
	- Apply ICA algorithm (e.g., FastICA)
	- Recover statistically independent components

## Code Example

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

## Advantages

- Excels in **blind source separation** tasks
- Extracts **statistically independent components**
- Effective when signals exhibit **non-Gaussian distributions**

## Disadvantages

- **Sensitive to noise and outliers**
- Assumes **number of sources ≤ number of observed signals**
- Relies on strong assumptions about **independence** and **non-Gaussianity**
	- May yield **unreliable components**
- **Computationally expensive**