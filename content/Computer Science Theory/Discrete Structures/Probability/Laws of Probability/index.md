---
title: Laws of Probability
---
> [!ABSTRACT]
> 
> This directory establishes the fundamental rules and axioms of probability theory. It covers how to quantify uncertainty, update beliefs based on new evidence, and analyze the relationships between different events.

---
## Knowledge Map
### Basic Counting & Axioms
- **[[Counting and Probability]]**: The starting point for discrete probability. It defines the probability of an event $E$ as the ratio of successful outcomes to the total sample space $|E|/|S|$ for uniform distributions.
### Relational Logic
- **[[Independent Events]]**: Explores scenarios where the occurrence of one event has no impact on the likelihood of another. This is the foundation for the product rule: $P(A \cap B) = P(A)P(B)$.
- **[[Conditional Probabilities]]**: Analyzes how the probability of an event "shifts" when we are given additional information or a restricted sample space.

### Inverse Probability
- **[[Bayes' Theorem]]**: A mathematical formula used to determine "inverse" conditional probabilities. It is the primary tool for updating the probability of a hypothesis as more evidence becomes available.

---
## Core Formulas

| **Theorem**             | **Context**                                                                          | **Formula**                            |
| ----------------------- | ------------------------------------------------------------------------------------ | -------------------------------------- |
| **Uniform Probability** | Used when all outcomes in the sample space are equally likely.                       | $$P(E) = \frac{\|E\|}{\|S\|}$$         |
| **Product Rule**        | The general rule for the probability of two events occurring together.               | $$P(A \cap B) = P(A \|B)P(B)$$         |
| **Independence**        | A simplified product rule valid **only** if $A$ and $B$ do not influence each other. | $$P(A \cap B) = P(A)P(B)$$             |
| **Bayes' Theorem**      | Used to calculate the probability of a cause ($F$) given an observed effect ($E$).   | $$P(F\|E) = \frac{P(E\|F)P(F)}{P(E)}$$ |
### Key Notation
- **$|E|$**: The number of outcomes in event $E$ (cardinality).
- **$|S|$**: The total number of outcomes in the sample space.
- **$P(A|B)$**: The probability of $A$ occurring **given** that $B$ has already occurred.
- **$A \cap B$**: The **intersection** (both events happening).

---
## Summary of Event Relationships
Understanding how events overlap is critical for choosing the right law:
- **Disjoint (Mutually Exclusive)**: The events cannot happen at the same time. $P(A \cap B) = 0$.
- **Independent**: The events can happen together, but one does not "inform" the other.
- **Dependent**: Knowing that one event occurred changes your estimate for the other.

---
## Related Toolkits
- **[[Computer Science Theory/Discrete Structures/Probability/Random Variables/index|Random Variables]]**: Once the laws are established, we apply them to variables that map outcomes to real numbers.
- **[[Computer Science Theory/Discrete Structures/Probability/Expected Values and Analysis/index|Expected Values and Analysis]]**: Uses these laws to calculate long-run averages (Expected Value).
- **[[Computer Science Theory/Discrete Structures/Counting/index|Counting]]**: Provides the permutations and combinations necessary to calculate $|E|$ and $|S|$ in complex scenarios.