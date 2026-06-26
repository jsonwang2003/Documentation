> [!INFO]
> Measures agreement between predicted and actual classifications while accounting for chance agreement.

## How It Works

Cohen’s Kappa compares observed accuracy with expected accuracy (random chance):

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

- $p_o$: Observed agreement  
- $p_e$: Expected agreement by chance  

## What to Look For

- Values range from -1 to 1  
- ($\kappa > 0.6$): Substantial agreement  
- Useful for multi-class classification and inter-rater reliability

## Application Models

- [[Random Forest Classification]]
- [[Support Vector Machine (SVM)]]
- [[Neural Network Regression]]