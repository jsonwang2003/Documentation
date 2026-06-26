> [!INFO]
> Measures the accuracy of probabilistic predictions in classification tasks. Lower values indicate better calibration.

## How It Works

Brier Score is the mean squared error of predicted probabilities:

$$
\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2
$$

- $f_i$: Predicted probability for instance $i$  
- $o_i$: Actual outcome (0 or 1)  
- $N$: Total number of predictions  

## What to Look For

- **Lower score = better calibrated predictions**  
- Useful for **probabilistic classifiers**  
- Sensitive to both **prediction confidence** and **correctness**

## Application Models

- [[Logistic Regression]]
- [[Random Forest Classification]]
- [[Gradient Boosting Machines (GBM)]]