> [!INFO]
> Measures the percentage error between actual and predicted values, useful for interpretability.

## How It Works

$$
\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

- Expresses error as a **percentage**  
- Undefined when $y_i = 0$

## What to Look For

- **Lower MAPE = better performance**  
- Easy to interpret across domains  
- Avoid when actual values can be zero

## Application Models

- [[Logistic Regression]]
- [[Time Series Analysis]]