
> [!INFO]
> A **supervised model** designed for **sequential data**, retaining temporal dependencies across time steps.

- Ideal for time-series forecasting and natural language tasks
## Components

- **Recurrent Units**: Maintain hidden state across time
- **Input Embedding**: Converts tokens or values into vector representations
- **Output Layer**: Generates predictions at each time step
- **Backpropagation Through Time (BPTT)**: Trains weights across sequences
## Key Features

1. **Temporal Memory**
	- Captures dependencies across time
2. **Sequential Prediction**
	- Outputs evolve with each time step
3. **Flexible Input Lengths**
	- Handles variable-length sequences
4. **Gradient-Based Learning**
	- Learns temporal patterns via BPTT
## Business Applications

- **Financial Forecasting**
	- Predicts stock trends from historical market data
- **Customer Behavior Modeling**
	- Analyzes transaction sequences to predict churn
- **Personalized Recommendations**
	- Suggests financial products based on user history
- **Risk Management**
	- Integrates forecasts into compliance workflows