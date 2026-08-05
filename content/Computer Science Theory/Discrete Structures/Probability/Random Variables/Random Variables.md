> [!ABSTRACT]
> 
> A Random Variable is a formal way to quantify outcomes. It is a function that maps every possible outcome in a sample space to a real number, allowing us to perform mathematical operations like averaging and summation on probabilistic events.

---
## Definition
A random variable $X$ is a function from the sample space $S$ to the set of real numbers $\mathbb{R}$:

$$
X: S \to \mathbb{R}
$$

### The Distribution
The distribution of a random variable $X$ (specifically the Probability Mass Function for discrete variables) is the function that assigns a probability to each possible value $r$:

$$
r \to P(X = r)
$$

> [!TIP]
> 
> Think of the distribution as the probability that an outcome results in a specific random variable value.

---
## Expectation
The **expectation** ($E[X]$), also known as the average or expected value, is the center of mass of the distribution. It can be calculated in two ways:
1. Summing over outcomes:
    $$
    E(X) = \sum_{s \in S} P(s)X(s)
    $$
    
2. Summing over values (The Distribution Method):
    $$
    E(X) = \sum_{r \in X(S)} P(X = r) \cdot r
    $$
    

---
## Example: Sum of Two Dice
Let $X$ be the random variable representing the **sum of the pips** of 2 fair dice.

![[Pasted image 20251019132329.png]]

- **Mapping**:
    - $X(5, 2) = 7$
    - $X(3, 3) = 6$
- **Distribution**: The likelihood of each sum varies based on the number of ways to achieve it.
    - $P(X = 7) = \frac{6}{36} = \frac{1}{6}$
    - $P(X = 9) = \frac{4}{36} = \frac{1}{9}$

---
## Types of Random Variables

|**Type**|**Description**|**Example**|
|---|---|---|
|**Discrete**|Values are countable (integers).|Number of heads in 10 flips.|
|**Continuous**|Values exist on a continuum (intervals).|The exact height of a person.|
|**Indicator**|Takes only values 0 or 1.|$X=1$ if a die is even, $X=0$ otherwise.|

---
## Related Notes
- **[[Distributions]]**: High-level overview of how probability mass is spread.
- **[[Expected Value]]**: Deeper dive into the properties of $E[X]$.
- **[[Linearity of Expectation]]**: A method to find the expectation of a sum of random variables without knowing their joint distribution.
- **[[Binomial Distribution]]**: A specific, famous discrete random variable distribution.