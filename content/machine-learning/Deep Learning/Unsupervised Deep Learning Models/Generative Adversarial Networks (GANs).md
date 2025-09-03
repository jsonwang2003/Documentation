
> [!INFO]
> A class of machine learning models where two neural networks — the **generator** and the **discriminator** — compete in a game-like setup to produce highly realistic synthetic data.

## Components

- **Generator**: Produce realistic synthetic data instances from random noise
- **Discriminator**: Differentiate between real and artificially generated outputs
## How it Works

- **Generator** improves by _learning to fool the discriminator_
- **Discriminator** gets better at _spotting fakes_
- Loop continues until the **generated data becomes indistinguishable from real data**
## Examples

- Businesses
	- Data augmentation
	- Synthetic data creation for privacy protection
	- Generate realistic multimedia content
- Fashion
	- Generate virtual clothing designs
	- Simulate fashion styles
		- Reducing cost and time associated with physical prototyping
- Finance
	- Fraudulent cases
		- Provided an alternative by identifying unusual patterns or behaviors in transaction data without prior fraud labels
		- Flags transactions that significantly deviate from typical encoded representations
		- Can generate synthetic yet realistic examples of fraudulent transactions to enhance the robustness of detection models