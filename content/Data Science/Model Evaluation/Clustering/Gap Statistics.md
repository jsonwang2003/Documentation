> [!INFO]
> Compares the performance of different clustering results using the expected dispersion under a reference null distribution.

## How It Works

$$
\text{Gap}(k) = E_n[\log(W_k^*)] - \log(W_k)
$$

- $W_k$: Within-cluster dispersion for $k$ clusters  
- $W_k^*$: Expected dispersion under null reference  
- $E_n$: Expectation over $n$ simulations  

## What to Look For

- **Higher gap = better clustering**  
- Helps determine optimal $k$  
- More robust than [[Elbow Method]]

## Application Models

- [[K-Means Clustering]] 
- [[Hierarchical Clustering]]