
> [!INFO]
> A type of neural network designed to learn efficient representations of data by compressing input into a lower-dimensional form (encoding) and then reconstructing it back (decoding) as closely as possible to the original.

- Represent one of the foundational unsupervised deep learning architectures
## Components:
- **Compression (Encoder)**: Transforms input data into a compact latent representation
- **Latent Space (Bottleneck)**: The compressed form that captures the most essential features
- **Decompression (Decoder)**: Reconstructs the original data from the latent representation
## How it Works

- uses [[machine-learning/Unsupervised Learning/index|Unsupervised Learning]], minimizing a **reconstruction loss** between the input and output
## Goal
- [[machine-learning/Unsupervised Learning/Dimensionality Reduction Models/index|Dimensionality Reduction]]: distill complex, high-dimensional data into a reduced, meaningful representation or latent space
## Reconstruction error
- Discrepancy between **original data** and **decoded reconstruction**
- When minimized will
	- Uncover intrinsic features
	- Enable businesses to process large datasets efficiently
- Variants enhances capability
	- Denoising autoencoders: facilitating noise removal
	- Variational autoencoders: probabilistic generation of new data