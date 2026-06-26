---
title: Random Variables
---
> [!ABSTRACT]
> 
> This directory covers the mathematical framework for mapping experimental outcomes to numerical values. It transitions from simple counting to functional analysis through probability distributions and specific statistical models.

---
## Knowledge Map
### Foundations
- **[[Random Variables]]**: The formal definition of $X: S \to \mathbb{R}$ and how to calculate the expectation $E[X]$ using outcomes or distribution values.
- **[[Distributions]]**: The study of how probability mass is allocated across a sample space and the application of the **Law of Total Probability**.
- **[[Uniform vs Non-Uniform Distributions]]**: Distinguishing between "fair" systems (equally likely) and "biased" or "weighted" systems.
### Specific Discrete Models
- **[[Binomial Distribution]]**: Models the number of successes in a fixed number of $n$ independent Bernoulli trials.
    - _Key formula:_ $P(k) = \binom{n}{k}p^k(1-p)^{n-k}$
- **[[Geometric Distribution]]**: Models the number of trials required to reach the **first** success in an infinite sequence.
    - _Key formula:_ $P(k) = (1-p)^{k-1}p$

---
## Quick Reference: Which Model to Use?

|**Scenario**|**Distribution**|**Sample Space**|
|---|---|---|
|Every outcome is equally likely|**Uniform**|Finite set $S$|
|Fixed number of trials, count successes|**Binomial**|$\{0, 1, \dots, n\}$|
|Repeat until first success occurs|**Geometric**|$\{1, 2, 3, \dots\}$|
|Outcomes have specific weighted odds|**Non-Uniform**|Any $S$|

---
## Core Identities
- **Sum of a Distribution**: $\sum_{a \in S} p(a) = 1$
- **Expectation of $X$**: $E[X] = \sum x \cdot P(X=x)$
- **Binomial Expectation**: $E[X] = np$
- **Geometric Expectation**: $E[X] = 1/p$

---
## Related Toolkits
- **[[Computer Science/Discrete Structures/Probability/Laws of Probability/index|Laws of Probability]]**: Provides the underlying logic for independence and conditional probability used in these distributions.
- **[[Computer Science/Discrete Structures/Probability/Expected Values and Analysis/index|Expected Value and Analysis]]**: Advanced techniques like **Linearity of Expectation** and **Case Analysis** used to solve complex problems involving these variables.