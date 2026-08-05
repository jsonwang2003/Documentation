> [!ABSTRACT]
> 
> A probability distribution defines how probability mass is allocated across a sample space $S$. It ensures that the likelihood of any specific outcome is between 0 and 1, and that the total likelihood of all possible outcomes sums exactly to 1.

---
## Probability Distributions
A distribution is a function $p: S \to [0, 1]$ such that:

$$
\sum_{a \in S} p(a) = 1
$$

### Examples of Discrete Distributions
- **Fair Coin**: $S = \{H, T\}$, where $p(H) = 1/2$ and $p(T) = 1/2$.
- **Fair Six-Sided Die**: $S = \{1, 2, 3, 4, 5, 6\}$, where $p(i) = 1/6$ for all $i$.
- **Standard Deck of Cards**: $S = \{A\spadesuit, K\spadesuit, \dots\}$, where $p(\text{card}) = 1/52$.

### Distribution of an Event
The probability of an event $E \subseteq S$ is the sum of the probabilities of the individual outcomes contained within that event:

$$
p(E) = \sum_{a \in E} p(a)
$$

- **Certainty**: $p(S) = 1$
- **Impossibility**: $p(\emptyset) = 0$

---
## The Law of Total Probability
The Law of Total Probability allows you to calculate the probability of an event $A$ by "weighting" it across different scenarios (a partition of the sample space).

![[Pasted image 20251018193715.png]]
### Partitioning with Complements
For any two events $A$ and $B$, you can calculate $P(A)$ by looking at cases where $B$ occurs and where it does not ($\overline{B}$):

$$
P(A) = P(A|B)P(B) + P(A|\overline{B})P(\overline{B})
$$
### General Form
If $\{B_1, B_2, \dots, B_k\}$ is a partition of the sample space $S$ (meaning they are mutually exclusive and their union is $S$), then for any event $A$:

$$
\boxed{P(A) = \sum_{i=1}^{k} P(A|B_i)P(B_i)}
$$

![[Pasted image 20251018201915.png]]

---
## Example: Maximum of Two Dice
**Question:** What is the probability that the maximum of two dice rolls is greater than 4?

Let $A$ be the event that $\max(\text{roll}_1, \text{roll}_2) > 4$. We can partition the space based on the first roll:
1. **Event $B$**: The first roll is $\{1, 2, 3, 4\}$.
    - $P(B) = 4/6 = 2/3$
    - $P(A|B)$: Given the first roll is $\leq 4$, the second roll must be $5$ or $6$ for the max to be $> 4$. Thus, $P(A|B) = 2/6 = 1/3$.
2. **Event $\overline{B}$**: The first roll is $\{5, 6\}$.
    - $P(\overline{B}) = 2/6 = 1/3$
    - $P(A|\overline{B})$: Since the first roll is already $> 4$, the condition is satisfied regardless of the second roll. Thus, $P(A|\overline{B}) = 1$.

Calculation:

$$
P(A) = \left(\frac{1}{3}\right)\left(\frac{2}{3}\right) + (1)\left(\frac{1}{3}\right) = \frac{2}{9} + \frac{3}{9} = \boxed{\frac{5}{9}}
$$

---
## Related Notes
- **[[Conditional Probabilities]]**: The Law of Total Probability is the denominator used in **[[Bayes' Theorem]]**.
- **[[Case Analysis]]**: This is the practical application of the Law of Total Probability to simplify complex problems.
- **[[Random Variables]]**: Distributions of events form the basis for Probability Mass Functions (PMFs).