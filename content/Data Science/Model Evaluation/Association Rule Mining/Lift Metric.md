> [!INFO]
> Assesses how much more likely a rule-based association is to occur compared to a random occurrence.

## How It Works

For a rule $X \rightarrow Y$:

$$
\text{Lift}(X \rightarrow Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X) \cdot \text{Support}(Y)}
$$

- Measures how much more likely $Y$ is to occur when $X$ occurs  
- Lift > 1: Positive association  
- Lift = 1: Independent  
- Lift < 1: Negative association

## What to Look For

- Lift > 1 indicates **meaningful association**  
- Helps filter out **spurious rules**  
- Use alongside **confidence** for stronger rule validation

## Application Models

- [[Apriori Algorithm]]
- [[Frequent Pattern Growth]]