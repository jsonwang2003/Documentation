> [!INFO]
> Modified R² that adjusts for the number of predictors in the model to prevent overfitting.

## How It Works

$$
\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - p - 1} \right)
$$

- $n$: Number of observations  
- $p$: Number of predictors  

Adjusted R² **penalizes unnecessary complexity**.

## What to Look For

- Use when comparing models with **different numbers of predictors**  
- Helps detect **overfitting**  
- Can be lower than [[R-Squared (R²)]]

## Application Models

- [[Multiple Linear Regression]]
- [[Classification Decision Tree]]
- [[Random Forest Classification]]