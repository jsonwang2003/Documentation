---
title: Expected Values and Analsis
---
> [!ABSTRACT]
> 
> This section focuses on long-run averages and the strategic tools used to break down and solve complex probabilistic systems. Once probabilities are established, analysis allows us to predict behavior and optimize decision-making.

---
## Knowledge Map
### Fundamentals of Expectation
- **[[Expected Value]]**: The foundational concept of the "weighted average" or center of mass of a distribution.
- **[[Linearity of Expectation]]**: A powerful property allowing the sum of expectations to be calculated without needing independence between variables.
### Advanced Analytical Tools
- **[[Conditional Expectation]]**: Calculating the average outcome given that a specific condition or event has already occurred.
- **[[Case Analysis]]**: The technique of partitioning a sample space into disjoint "cases" to simplify a larger problem.
- **[[Random Sampling]]**: Methods for selecting outcomes (Uniform, Rejection) to simulate distributions or study populations.

---
## Core Principles
- Expectation of a Random Variable
    $$
    E[X] = \sum_{x \in X(S)} x \cdot P(X=x)
    $$
    
- Linearity of Expectation
    $$
    E[X + Y] = E[X] + E[Y]
    $$
    
    (Valid even if $X$ and $Y$ are dependent)
    
- Law of Total Expectation
    $$
    E[X] = E[X|A]P(A) + E[X|\overline{A}]P(\overline{A})
    $$
    
- Expectation of an Indicator Variable
    For an event $A$, let $I_A$ be 1 if $A$ occurs and 0 otherwise:
    
    $$
    E[I_A] = P(A)
    $$
    

---
## Problem Solving Strategies
1. **Indicator Method**: Break a complex random variable into a sum of simpler $\{0, 1\}$ indicator variables and apply Linearity of Expectation.
2. **First-Step Analysis**: Use Conditional Expectation and Case Analysis to set up recursive equations (common in **[[Geometric Distribution]]** problems).
3. **Rejection Method**: Use Random Sampling to generate outcomes from a difficult distribution by "filtering" a simpler one.

---
## Related Toolkits
- **[[Computer Science/Discrete Structures/Probability/Laws of Probability/index|Laws of Probability]]**: Provides the conditional logic and independence rules used in analysis.
- **[[Random Variables/index|Random Variables]]**: Defines the distributions (Binomial, Geometric) whose expectations we analyze here.