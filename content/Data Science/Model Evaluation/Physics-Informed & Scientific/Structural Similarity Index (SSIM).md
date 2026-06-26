> [!INFO]
> Evaluates the perceptual similarity of generated physical field data, used in scientific computing.

## How It Works

$$
\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

- $\mu_x$, $\mu_y$: Mean values of images $x$ and $y$  
- $\sigma_x^2$, $\sigma_y^2$: Variances  
- $\sigma_{xy}$: Covariance  
- $C_1$, $C_2$: Stabilizing constants  

SSIM compares **structural features** rather than **pixel-wise differences**

## What to Look For

- Range: $-1$ to $+1$  
- **Higher SSIM = better perceptual similarity**  
- Ideal for evaluating spatial fields, simulations, and reconstructions

## Application Models

- [[physics-informed/physics-based-neural-networks]]  
- [[physics-informed/image-based-simulation-models]]