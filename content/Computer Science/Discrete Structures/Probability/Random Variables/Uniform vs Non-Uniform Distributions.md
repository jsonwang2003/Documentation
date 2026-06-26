> [!ABSTRACT]
> 
> This note distinguishes between distributions where outcomes are equally likely and those where probability mass is distributed unevenly. This distinction determines whether simple counting methods or weighted calculations must be used.

---
## Uniform Distribution

> [!INFO]
> 
> A distribution is Uniform if it assigns the same probability to every outcome in the sample space $S$.

Because the sum of all probabilities must equal $1$, the probability of any single outcome $a$ is:

$$
p(a) = \frac{1}{|S|}
$$

### Examples
- **Fair Coin Flip**: $S = \{H, T\} \implies p(H) = 1/2, p(T) = 1/2$.
- **Fair Six-sided Die**: $S = \{1, 2, 3, 4, 5, 6\} \implies p(i) = 1/6$ for each $i$.
- **Standard Deck of Cards**: $S = \{A\spadesuit, K\spadesuit, \dots\} \implies p(\text{any specific card}) = 1/52$.

---
## Non-Uniform Distribution

> [!INFO]
> 
> A distribution is Non-Uniform if at least two outcomes in the sample space have different probabilities.

In these cases, you cannot simply count outcomes ($|E|/|S|$) to find probability; you must sum the specific weights assigned to each outcome in the event.
### Examples
1. **Biased Coin**: A coin where $p(H) = 2/3$ and $p(T) = 1/3$.
2. **Weighted Die**: A die where faces are not equally likely:
    - $p(1) = p(2) = p(3) = p(4) = 1/8$
    - $p(5) = p(6) = 1/4$
    - _(Note: The sum is still $4(1/8) + 2(1/4) = 0.5 + 0.5 = 1$)_.

---
## Comparison Table

| **Feature**            | **Uniform Distribution**                                   | **Non-Uniform Distribution**                         |
| ---------------------- | ---------------------------------------------------------- | ---------------------------------------------------- |
| **Probability $p(a)$** | Every outcome has the probability $p(a) = \frac{1}{\|S\|}$ | Different outcomes can have different probabilities. |
| **Calculation Method** | $$P(E) = \frac{\|E\|}{\|S\|}$$                             | $$P(E) = \sum_{a \in E} p(a)$$                       |
| **Typical Context**    | "Fair", "Randomly selected"                                | "Biased", "Weighted," "Empirical"                    |

---
## Important Considerations
- **Assumptions**: Many introductory problems assume a uniform distribution (e.g., "a random 4-digit string"). If the problem does not specify "uniform" or "fair," check for given weights.
- **Random Variables**: A uniform distribution on a sample space (like die rolls) can result in a **non-uniform** distribution for a random variable (like the _sum_ of two dice).