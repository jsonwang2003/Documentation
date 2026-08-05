> [!ABSTRACT]
> 
> The Geometric Distribution models the number of independent Bernoulli Trials required to achieve the first success. It is defined over an infinite discrete sample space.

---
## Definition
Suppose we conduct a sequence of independent trials, each with a probability of success $p$ and probability of failure $q = (1-p)$. The random variable $X$ represents the number of the trial on which the **first success** occurs.
### Probability Mass Function (PMF)
For $k \in \{1, 2, 3, \dots\}$:

$$
\boxed{f(k) = P(X = k) = (1-p)^{k-1}p}
$$

- **$(1-p)^{k-1}$**: The probability of having $k-1$ consecutive failures.
- **$p$**: the probability that the $k^{th}$ trial is a success.

---
## Proof: Sum of Probabilities
By the **[[Distributions#Law of Total Probability|Law of Total Probability]]**, the sum of all probabilities in the sample space $S = \mathbb{Z}^+$ must equal $1$.

Using the formula for an infinite **Geometric Series** where the first term $a_1 = 1$ and the common ratio $r = (1-p)$:

$$
\begin{align*} 
\sum_{k=1}^\infty (1-p)^{k-1} \cdot p &= p \cdot \sum_{k=1}^\infty (1-p)^{k-1} \\
&= p \cdot \left(\frac{1}{1-(1-p)}\right) \\
&= p \cdot \left(\frac{1}{p}\right) \\
&= \boxed{1} 
\end{align*}
$$

> [!NOTE]
> 
> The series converges because $0 < p < 1$, which implies $|1-p| < 1$.

---
## Properties

|**Property**|**Value**|
|---|---|
|**Sample Space**|$S = \{1, 2, 3, \dots\}$|
|**Expected Value**|$E[X] = \frac{1}{p}$|
|**Memoryless Property**|$P(X > n+k \mid X > n) = P(X > k)$|

### The Memoryless Property

The Geometric distribution is unique among discrete distributions because it is **memoryless**. This means that if you have already failed $n$ times, the probability of needing $k$ more trials is the same as the probability of needing $k$ trials at the very start. The "coin has no memory" of previous failures.

---
## Example: Rolling a Die
What is the expected number of rolls to get a **6** on a fair die?
- Here, $p = 1/6$.
- Using the expectation formula: $E[X] = \frac{1}{1/6} = \boxed{6}$ rolls.

---
## Related Notes
- **[[Binomial Distribution]]**: Contrast this with the Binomial, where the number of trials $n$ is fixed and we count successes.
- **[[Case Analysis]]**: Often used to derive the Expected Value of a geometric distribution via "First-Step Analysis."
- **[[Expected Value]]**: The long-run average of trials needed for success.