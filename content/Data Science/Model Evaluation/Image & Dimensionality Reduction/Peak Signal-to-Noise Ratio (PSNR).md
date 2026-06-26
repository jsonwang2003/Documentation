> [!INFO]
> Measures the quality of reconstructed images compared to the original, used in image processing.

## How It Works

PSNR is derived from the [[Mean Squared Error (MSE)]] between the original image $I$ and the reconstructed image $K$:

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{\text{MSE}} \right)
$$

- $MAX_I$: Maximum possible pixel value (e.g., 255 for 8-bit images)  
- $\text{MSE}$: Mean squared error between $I$ and $K$  

Higher PSNR indicates better reconstruction fidelity.

## What to Look For

- Typical range: 30–50 dB for 8-bit images  
- **Higher PSNR = lower distortion**  
- Sensitive to **pixel-level differences**, but **not perceptual quality**  
- Best used when comparing similar codecs or reconstruction methods

## Application Models

- [[Autoencoders]]
- [[Generative Adversarial Networks (GANs)]]
- [[Convolutional Neural Network (CNN)]]