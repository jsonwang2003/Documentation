> [!INFO]
> Square root of MSE, making it interpretable in the same unit as the target variable. Penalizes large errors.

## How It Works

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 }
$$

- Same formula as [[Mean Squared Error (MSE)]], but square root applied  
- Easier to interpret than [[Mean Squared Error (MSE)]]

## What to Look For

- **Lower RMSE = better performance**  
- More sensitive to large errors than [[Mean Absolute Error (MAE)]]  
- Use when **error magnitude matters**

## Application Models

- [[Logistic Regression]]
- [[Classification Decision Tree]]
- [[Random Forest Classification]]
- [[Neural Network Regression]]