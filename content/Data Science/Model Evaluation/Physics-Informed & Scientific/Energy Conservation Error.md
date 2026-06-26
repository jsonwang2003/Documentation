> [!INFO]
> Assesses how well a model conserves energy in simulations, ensuring physical consistency.

## How It Works

$$
\text{Energy Error} = |E_{\text{pred}} - E_{\text{true}}|
$$

- $E_{\text{pred}}$: Predicted total energy  
- $E_{\text{true}}$: True or expected conserved energy  

This metric checks whether the model violates [[conservation laws]] over time or space.

## What to Look For

- **Should be near zero** for physically consistent models  
- Critical in **Fluid Dynamics**, [[Physics/Thermodynamics/index|Thermodynamics]], and [[Physics/Mechanics/index|Mechanical Systems]]  
- Often tracked across simulation steps

## Application Models

- [[physics-informed/pinns]]  
- [[physics-informed/simulation-driven-models]]