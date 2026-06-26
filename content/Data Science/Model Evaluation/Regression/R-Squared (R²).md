> [!INFO]
> Represents the proportion of variance explained by a regression model, assessing its fit.

## How It Works

$$
R^2 = 1 - \frac{ \sum (y_i - \hat{y}_i)^2 }{ \sum (y_i - \bar{y})^2 }
$$

- $\bar y$: Mean of actual values  
- $R^2$ ranges from 0 to 1 (or negative if model is worse than baseline)

## What to Look For

- **Higher $R^2$ = better fit**  
- Can be misleading with **non-linear models or overfitting**  
- Use alongside [[residual plots]] for deeper insight

## Application Models

- [[Deep Neural Network (DNN)]]  
- [[Recurrent Neural Network (RNN)]]