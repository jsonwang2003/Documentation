> [!INFO]
> Quantifies the error in governing equations, used in physics-informed neural networks (PINNs).

## How It Works

$$
\text{Residual Loss} = \frac{1}{n} \sum_{i=1}^{n} \left| \mathcal{N}(u_i) - f_i \right|^2
$$

- $\mathcal{N}(u_i)$: Differential operator applied to predicted solution  
- $f_i$: Known forcing term or target  
- $n$: Number of collocation points  

This measures how well the model satisfies the underlying PDEs.

## What to Look For

- **Lower residual = better physics compliance**  
- Use alongside data loss for balanced training  
- Can be decomposed by equation type (e.g., momentum, continuity)

## Application Models

- [[physics-informed/pinns]]  
- [[physics-informed/pde-solvers]]