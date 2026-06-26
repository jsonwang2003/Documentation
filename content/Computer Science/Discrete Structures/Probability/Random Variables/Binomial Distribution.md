> [!ABSTRACT]
> 
> The Binomial Distribution models the number of successes in a fixed number of independent Bernoulli Trials, each with the same probability of success $p$. Unlike a uniform distribution, the outcomes are not necessarily equally likely.

---
## Bernoulli Trial
A **Bernoulli Trial** is a performance of an experiment with exactly **two possible outcomes** (e.g., flipping a coin, a part being defective or non-defective).
- **Success** with probability $p$.
- **Failure** with probability $1 - p$.

---
## Binomial Distribution Formula
For a particular number of trials $n$ and probability $p$, the sample space is the set of integers $\{0, 1, 2, \dots, n\}$. The probability of achieving exactly $k$ successes is:

$$
\boxed{P(k) = \binom{n}{k}p^k(1-p)^{n-k}}
$$

### Understanding the Components
- **$\binom{n}{k}$**: The number of ways to choose which $k$ trials out of $n$ result in success.
- **$p^k$**: The probability that $k$ specific trials result in success.
- **$(1-p)^{n-k}$**: The probability that the remaining $n-k$ trials result in failure.

---
## Examples
### 1. Fair Coins (Uniform Case)
When flipping $n$ fair coins, the probability of getting exactly $k$ Heads ($H$) is:

$$
P(k\text{ Hs}) = \frac{\binom{n}{k}}{2^n}
$$

> [!NOTE]
> 
> Here, $p = 0.5$ and $(1-p) = 0.5$. Since $0.5^k \cdot 0.5^{n-k} = 0.5^n$, which is $1/2^n$, the formula simplifies to the ratio of successful sequences over total possible sequences ($2^n$).

### 2. Biased Trials (Non-Uniform Case)
What if the coin isn't fair? If a biased coin has $P(H) = 0.6$ and you flip it 10 times, the probability of getting exactly 7 Heads is:

$$
P(7) = \binom{10}{7}(0.6)^7(0.4)^3
$$

---
## Properties and Analysis

|**Property**|**Formula**|
|---|---|
|**Sample Space**|$S = \{0, 1, 2, \dots, n\}$|
|**Expected Value**|$E[X] = np$|
|**Variance**|$Var(X) = np(1-p)$|

---
## Related Notes
- **[[Independent Events]]**: Trials must be independent for this distribution to apply.
- **[[Expected Value]]**: The average number of successes over $n$ trials.
- **[[Linearity of Expectation]]**: Used to easily derive $E[X] = np$ by summing $n$ indicator variables.
- **[[Geometric Distribution]]**: Contrast this with the distribution where the number of trials is not fixed, but continues until the first success.