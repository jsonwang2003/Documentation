> [!INFO]
> Measures the deviation between predicted and actual physical quantities relative to actual values.

## How It Works

$$
\text{Relative Error} = \frac{|y - \hat{y}|}{|y|}
$$

- $y$: Actual physical value  
- $\hat{y}$: Predicted value  

This metric expresses the error as **a fraction of the true value**, making it **scale-aware**.

## What to Look For

- Lower values indicate **better physical fidelity**  
- Useful when **absolute error is less meaningful due to varying scales**  
- Sensitive to small denominators &rarr; **avoid when $y \approx 0$**

## Application Models

- [[physics-informed/pinns]]  
- [[physics-informed/scientific-ml-models]]