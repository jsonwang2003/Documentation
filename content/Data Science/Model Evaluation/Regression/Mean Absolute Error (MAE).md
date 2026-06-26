> [!INFO]
> Measures the average absolute differences between predicted and actual values in regression tasks.

## How It Works

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$

- $y_i$: Actual value  
- $\hat{y}_i$: Predicted value  
- $n$: Number of data points  

MAE gives a straightforward **average of prediction errors**, treating all deviations equally.

## What to Look For

- **Lower MAE = better performance**  
- Easy to interpret  
- Less sensitive to outliers than [[Mean Squared Error (MSE)]] or [[Root Mean Squared Error (RMSE)]]

## Application Models

- [[Deep Neural Network (DNN)]]  
- [[Recurrent Neural Network (RNN)]]