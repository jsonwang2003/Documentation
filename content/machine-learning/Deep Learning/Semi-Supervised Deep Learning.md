> [!INFO]
> Semi-supervised deep learning combines supervised and unsupervised techniques to **leverage both labeled and unlabeled data**, enabling scalable learning when labeled data is scarce.

## Process

- Initial training on labeled data
- Iterative refinement using unlabeled data
- Common techniques:
	- [[Pseudo-labeling]]
	- [[Consistency regularization]]
	- [[Generative modeling]]
- Applied in domains such as:
	- Healthcare diagnostics
	- Natural language understanding
	- Image classification and segmentation

## Advantages

- Reduces labeling cost and manual effort
- Improves generalization and model robustness
- Scales effectively with **growing unlabeled datasets**
- Balances accuracy and training efficiency

## Disadvantages

- Sensitive to quality of pseudo-labels
- Requires careful tuning of **regularization strategies**
- May propagate errors from mislabeled or noisy data
- Evaluation can be complex due to mixed supervision