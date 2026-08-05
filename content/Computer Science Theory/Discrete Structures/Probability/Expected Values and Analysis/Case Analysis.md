> [!ABSTRACT]
> 
> Case analysis is a problem-solving strategy used to calculate probabilities or expected values by partitioning a complex problem into smaller, mutually exclusive, and collectively exhaustive scenarios (cases).

---
## The Core Concept
Case analysis relies on the **Law of Total Probability** and the **Law of Total Expectation**. By breaking a sample space $S$ into disjoint cases $C_1, C_2, \dots, C_n$, you can solve for each case individually and then aggregate the results.
### Conditions for Cases
To ensure the analysis is mathematically sound, the chosen cases must be:
- **Mutually Exclusive**: No two cases can happen at the same time ($C_i \cap C_j = \emptyset$).
- **Collectively Exhaustive**: The cases must cover every possible outcome of the experiment ($\bigcup C_i = S$).

---
## Probability by Case Analysis
When calculating the probability of an event $A$, you can use the **Law of Total Probability**:

$$
P(A) = \sum_{i=1}^{n} P(A | C_i)P(C_i)
$$

### Example: Selecting a Coin
Suppose you have two jars. Jar 1 has 3 gold coins and 1 silver coin. Jar 2 has 1 gold coin and 3 silver coins. You pick a jar at random (50% chance each) and then pick a coin. What is the probability of picking a gold coin ($G$)?
- **Case 1 ($C_1$):** You pick Jar 1. $P(C_1) = 0.5$, $P(G | C_1) = 0.75$.
- **Case 2 ($C_2$):** You pick Jar 2. $P(C_2) = 0.5$, $P(G | C_2) = 0.25$.
- **Result**: $P(G) = (0.75 \cdot 0.5) + (0.25 \cdot 0.5) = 0.5$.

---
## Expectation by Case Analysis
Similarly, the **Law of Total Expectation** allows you to calculate the expected value $E[X]$ by weighting the expectation within each case by the probability of that case occurring:

$$
E[X] = \sum_{i=1}^{n} E[X | C_i)P(C_i)
$$

This is particularly useful for problems involving **[[Computer Science Theory/Discrete Structures/Discrete Algorithms/Recursive Algorithms/index|Recursive Algorithms]]** or state-based transitions, such as finding the expected number of trials in a **[[Geometric Distribution]]**.

---
## When to Use Case Analysis

|**Scenario**|**Application**|
|---|---|
|**Multi-stage processes**|Use cases to represent different paths in a decision tree.|
|**Unknown Parameters**|Use cases when the probability of an event depends on an unknown prior condition.|
|**Complex Sample Spaces**|Use cases to simplify "at least" or "at most" problems by looking at disjoint counts.|

---
## Related Notes
- **[[Conditional Probabilities]]**: The foundation of the "if-then" logic used in defining cases.
- **[[Conditional Expectation]]**: The mathematical formalization of $E[X | C_i]$.
- **[[Bayes' Theorem]]**: Often used after a case analysis to "reverse" the condition (e.g., given we found a gold coin, what is the probability it came from Jar 1?).
- **[[Geometric Distribution]]**: Frequently solved using case analysis on the first trial (Success vs. Failure).