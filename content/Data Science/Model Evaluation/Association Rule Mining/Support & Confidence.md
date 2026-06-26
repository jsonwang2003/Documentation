> [!INFO]
> Evaluates the strength of association rules based on the proportion of transactions that contain both antecedent and consequent items.

## How It Works

For a rule $X \rightarrow Y$:

- **Support**:  
  $$
  \text{Support}(X \rightarrow Y) = \frac{\text{Transactions containing } X \cup Y}{\text{Total transactions}}
  $$

- **Confidence**:  
  $$
  \text{Confidence}(X \rightarrow Y) = \frac{\text{Transactions containing } X \cup Y}{\text{Transactions containing } X}
  $$

Support measures **how frequently the rule occurs**.  
Confidence measures **how reliably $Y$ occurs when $X$ does**.

## What to Look For

- **High support = frequent pattern**  
- **High confidence = strong conditional dependency**  
- Use *thresholds* to prune weak or rare rules

## Application Models

- [[Apriori Algorithm]]
- [[Equivalence Class Clustering and Bottom-Up Lattice Traversal (ECLAT)]]