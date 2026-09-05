> [!ABSTRACT]
> 
> The expected value ($E[X]$) is the long-run average value of a random variable over many repetitions of an experiment. It is the probability-weighted average of all possible outcomes.

---
## Formal Definition
The expectation of a random variable $X$ on a sample space $S$ can be calculated in two equivalent ways:
1. By Outcome: Summing over every individual outcome $s$ in the sample space $S$.
    $$
    E(X) = \sum_{s \in S}P(s)X(s)
    $$
    
2. By Value: Summing over every possible value $r$ that the random variable $X$ can take.
    $$
    E(X) = \sum_{r \in X(S)} P(X=r) \cdot r
    $$
    

---
## Examples
### 1. Rolling a Single Die
Let $X$ be the number showing on a fair six-sided die after one roll. Each outcome has a probability of $1/6$.

$$
\begin{align*} 
E(X) &= 1 \cdot P(X=1) + 2 \cdot P(X=2) + 3 \cdot P(X=3) + 4 \cdot P(X=4) + 5 \cdot P(X=5) + 6 \cdot P(X=6) \\ 
&= 1 \cdot \left(\frac{1}{6}\right) + 2 \cdot \left(\frac{1}{6}\right) + 3 \cdot \left(\frac{1}{6}\right) + 4 \cdot \left(\frac{1}{6}\right) + 5 \cdot \left(\frac{1}{6}\right) + 6 \cdot \left(\frac{1}{6}\right) \\
&= \boxed{3.5} 
\end{align*}
$$

### 2. Sum of Two Dice
Let $X$ be the random variable representing the **sum of pips** after rolling two fair dice. The possible values for $X$ range from $2$ to $12$.

![[Pasted image 20251019132329.png]]

To find the expectation, we multiply each possible sum by its corresponding probability:

$$
\begin{align*} 
E(X) &= 2\left(\frac{1}{36}\right) + 3\left(\frac{1}{18}\right) + 4\left(\frac{1}{12}\right) + 5\left(\frac{1}{9}\right) + 6\left(\frac{5}{36}\right) + 7\left(\frac{1}{6}\right) + 8\left(\frac{5}{36}\right) + 9\left(\frac{1}{9}\right) + 10\left(\frac{1}{12}\right) + 11\left(\frac{1}{18}\right) + 12\left(\frac{1}{36}\right) \\
&= \boxed{7} 
\end{align*}
$$

---
## Properties and Applications

|**Concept**|**Application**|
|---|---|
|**[[Linearity of Expectation]]**|A shortcut for the two-dice problem: $E[X_1 + X_2] = E[X_1] + E[X_2] = 3.5 + 3.5 = 7$.|
|**[[Conditional Expectation]]**|Used when calculating the average outcome given that a certain event has already occurred.|
|**[[Distributions]]**|Every standard distribution (Binomial, Geometric, etc.) has a predefined formula for its expected value.|

---
## Related Notes
- **[[Random Variables Definition]]**: Expected value is only defined for numerical random variables.
- **[[Conditional Expectation#Law of Total Expectation|Law of Total Expectation]]**: A method for calculating $E[X]$ by partitioning the sample space into cases.
- **[[Case Analysis]]**: Useful for finding the expected value of complex, multi-stage experiments.