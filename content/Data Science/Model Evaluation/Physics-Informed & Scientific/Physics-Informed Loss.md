> [!INFO]
> Incorporates physical constraints into loss functions to ensure model adherence to known laws.

## How It Works

$$
\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{physics}}
$$

- $\mathcal{L}_{\text{data}}$: Error between predictions and observed data  
- $\mathcal{L}_{\text{physics}}$: Residual loss from governing equations  
- $\lambda$: Weighting factor for physics enforcement  

This composite loss ensures the model learns both data fidelity and physical consistency.

## What to Look For

- **Balance between data and physics terms is crucial**  
- Adjust $\lambda$ to **control trade-off**  
- Use domain-specific constraints for better generalization

## Application Models

- [[physics-informed/pinns]]