---
title: Probability
---
> [!ABSTRACT]
> 
> Probability is the mathematical study of uncertainty and the likelihood of events occurring. This directory covers the transition from basic counting principles to complex distributions and analytical tools used to quantify random processes.

---
## Knowledge Map
## [[Computer Science/Discrete Structures/Probability/Laws of Probability/index|Laws of Probability]]
- Focuses on the **fundamental rules and logic** used to calculate the likelihood of events within a sample space.
- Covers **Foundations** (Uniform Probability, Sample Spaces), **Independence**, and **Conditional Probabilities**.
- Includes **Bayes' Theorem** as a primary tool for updating probabilities based on new evidence or observed data.
- Essential for establishing the formal axioms required for rigorous statistical analysis.
## [[Random Variables/index|Random Variables]]
- Explores the **functional mapping** of experimental outcomes to numerical values $X: S \to \mathbb{R}$.
- Categorizes behavior through **Distributions**, distinguishing between **Uniform** (equally likely) and **Non-Uniform** (weighted) scenarios.
- Details specific discrete models like the **Binomial Distribution** (counting successes) and the **Geometric Distribution** (trials until success).
- Provides the structure needed to move from qualitative outcomes to quantitative data analysis.
## [[Computer Science/Discrete Structures/Probability/Expected Values and Analysis/index|Expected Values and Analysis]]
- Targets the **long-run average behavior** and strategic decomposition of probabilistic systems.
- Leverages the **Linearity of Expectation** to solve complex sums and **Conditional Expectation** to refine predictions.
- Utilizes **Case Analysis** to partition sample spaces into manageable disjoint scenarios for solving multi-stage problems.
- Bridges theoretical distributions with practical applications in **Random Sampling** and algorithmic performance.

---
## Distribution Comparison

|**Distribution**|**Type**|**Key Characteristic**|**Typical Use Case**|
|---|---|---|---|
|**Uniform**|Discrete|Every outcome has equal probability.|Rolling a fair die.|
|**Binomial**|Discrete|Fixed $n$ trials, constant $p$ success.|Number of heads in 10 flips.|
|**Geometric**|Discrete|Number of trials _until_ first success.|Flips needed to get the first head.|

---
## Core Theorems
### Bayes' Theorem
Used to update the probability of a hypothesis ($A$) given the presence of evidence ($B$).

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

### Linearity of Expectation
The expectation of a sum of random variables is equal to the sum of their individual expectations, regardless of whether they are independent.

$$
E\left[\sum_{i=1}^{n} X_i\right] = \sum_{i=1}^{n} E[X_i]
$$

---
## Related Toolkits
- **[[Computer Science/Discrete Structures/Counting/index|Counting]]**: Combinatorial foundations for building sample spaces.
- **[[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/index|Recursive Algorithms]]**: Essential for analyzing "state-based" problems and trials-until-success scenarios.
- **[[Asymptotic Notation]]**: Used to analyze the behavior of distributions and error bounds as $n \to \infty$.