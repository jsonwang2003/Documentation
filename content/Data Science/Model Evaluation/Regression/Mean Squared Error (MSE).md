> [!INFO]
> Measures the average squared difference between actual and predicted values, used for regression models.

## How It Works

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
$$

- Penalizes larger errors more than [[Mean Absolute Error (MAE)]]  
- Sensitive to outliers

## What to Look For

- **Lower MSE = better performance**  
- Use when _large errors are especially undesirable_  
- Not directly interpretable in target units

## Application Models

- [[Deep Neural Network (DNN)]]  
- [[Recurrent Neural Network (RNN)]]