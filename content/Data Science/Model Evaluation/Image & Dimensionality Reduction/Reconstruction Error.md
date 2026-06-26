> [!INFO]
> Measures the difference between original data and reconstructed data, often used in dimensionality reduction methods like PCA and autoencoders.

## How It Works

For input $x$ and its reconstruction $\hat{x}$:

$$
\text{Reconstruction Error} = \frac{1}{n} \sum_{i=1}^{n} \left\| x_i - \hat{x}_i \right\|^2
$$

- $x_i$: Original data point  
- $\hat{x}_i$: Reconstructed data point  
- $n$: Number of samples  

This metric captures how well the model preserves the original structure.

## What to Look For

- **Lower error = better reconstruction**  
- Useful for [[Anomaly Detection]] and compression quality  
- Can be visualized as residual maps or error histograms

## Application Models

- [[Principal Component Analysis (PCA)]]
- [[Autoencoders]]