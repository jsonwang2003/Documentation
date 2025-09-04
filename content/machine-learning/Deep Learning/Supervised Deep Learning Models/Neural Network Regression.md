
> [!INFO]
> A **supervised learning model** designed to predict **continuous numerical values** using a **feedforward neural network** architecture.

- Ideal for tasks like price prediction, demand forecasting, and risk scoring
- Learns complex, non-linear relationships between input features and target outputs
## Components

- **Input Layer**
	- Accepts structured features (e.g., numerical, categorical embeddings)
- **Hidden Layers**
	- Perform non-linear transformations using activation functions (e.g., ReLU, tanh)
- **Output Layer**
	- Typically a single neuron with **linear activation** for continuous output
- **Loss Function**
	- Common choices: **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**
- **Optimizer
	- Gradient-based methods like **Adam**, **SGD** for weight updates
- **Regularization**
	- Techniques like **dropout** or **L2 penalty** to prevent overfitting
## Key Features

1. **Continuous Output Prediction**
	- Unlike classification, outputs are real-valued (e.g., prices, temperatures)
2. **Non-Linear Modeling**
	- Captures complex relationships beyond [[data-science/Regression Analysis/Regression Models#Linear Regression|linear regression]]
3. **Flexible Architecture**
	- Depth and width can be tuned for task complexity
4. **End-to-End Learning**
	- Learns directly from raw or engineered features to output
5. **Scalable to High-Dimensional Data**
	- Handles large feature sets with appropriate regularization
## Business Applications

- **Real Estate**
	- Predict housing prices from features like size, location, amenities
- **Finance**
	- Forecast stock prices, credit risk scores, or loan default probabilities
- **Retail**
	- Estimate future demand or sales volume based on historical data
- **Energy**
	- Predict consumption patterns for load balancing and cost optimization
- **Healthcare**
	- Model patient outcomes or treatment costs from clinical data