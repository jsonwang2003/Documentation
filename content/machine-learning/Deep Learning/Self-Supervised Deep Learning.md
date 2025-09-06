> [!INFO]
> **Self-supervised learning** operates entirely on **unlabeled data**, generating labels from intrinsic data properties. Models learn via **proxy tasks** that uncover meaningful representations, enabling scalable pretraining across domains.

## Process

- No human-provided labels required
- Learns through proxy tasks such as:
	- **Predicting rotations**
	- **Reconstructing masked inputs**
	- **Forecasting future states**
- Typically involves pretraining followed by fine-tuning on downstream tasks
- Applied in:
	- **Image classification**
	- **Video analysis**
	- **Language modeling**
	- **Recommendation systems** (e.g., Netflix, Spotify)

## Advantages
- **Eliminates manual labeling effort**
- Produces **transferable, generalizable representations**
- Ideal for **large-scale pretraining** on user-generated data
- **Reduces privacy risks** and supports **continuous model improvement**

## Disadvantages
- Proxy tasks may not align perfectly with downstream objectives
- Requires careful design to avoid **trivial solutions**
- Evaluation can be complex due to lack of explicit supervision
- Fine-tuning may still require labeled data for task-specific performance
