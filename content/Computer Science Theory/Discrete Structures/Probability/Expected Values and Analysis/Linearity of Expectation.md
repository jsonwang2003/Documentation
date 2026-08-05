> [!ABSTRACT]
> 
> Linearity of Expectation is a powerful property of expected values which states that the expectation of a sum of random variables is equal to the sum of their individual expectations. Crucially, this property holds true even if the random variables are dependent.

---
## The Theorem
For any finite collection of random variables $X_1, X_2, \dots, X_n$, the expected value of their sum is:

$$
\boxed{E[X_1 + X_2 + \dots + X_n] = E[X_1] + E[X_2] + \dots + E[X_n]}
$$

### General Form
In its most general linear form, including constants $a$ and $b$:

$$
E[aX + bY] = aE[X] + bE[Y]
$$

---
## Key Property: No Independence Required
The most significant aspect of Linearity of Expectation is that it **does not require the variables to be independent**. While the probability of a joint event $P(A \cap B)$ changes based on dependence, the average of their sum remains additive.
- **To find $P(A \cap B)$**: You must know if $A$ and $B$ are independent.
- **To find $E[X + Y]$**: You only need to know the individual expectations $E[X]$ and $E[Y]$.

---
## Examples
### 1. Sum of Two Dice
In the [[Expected Value]] note, we calculated the sum of two dice by summing all 36 possible outcomes, resulting in $7$. Using Linearity of Expectation, we can simplify this significantly:

Let $X_1$ be the result of the first die and $X_2$ be the result of the second die.
- We know $E[X_1] = 3.5$ and $E[X_2] = 3.5$.
- By linearity: $E[X_1 + X_2] = E[X_1] + E[X_2]$
- $3.5 + 3.5 = \boxed{7}$

### 2. Indicator Variables (Binomial Expectation)
Suppose you flip a biased coin $n$ times where the probability of heads is $p$. Let $X$ be the total number of heads. We can define an **indicator variable** $X_i$ for each flip:
- $X_i = 1$ if the $i^{th}$ flip is heads.
- $X_i = 0$ if the $i^{th}$ flip is tails.

The expectation of one indicator variable is $E[X_i] = (1 \cdot p) + (0 \cdot (1-p)) = p$.

The total number of heads is $X = X_1 + X_2 + \dots + X_n$.

By linearity:

$$
E[X] = \sum_{i=1}^{n} E[X_i] = \sum_{i=1}^{n} p = \boxed{np}
$$

---
## Related Notes
- **[[Expected Value]]**: The fundamental definition of the weighted average.
- **[[Random Variables]]**: The numerical functions upon which expectation is calculated.
- **[[Independent Events]]**: While not required for linearity, independence is required for the _variance_ of a sum to be additive.
- **[[Binomial Distribution]]**: Where the $E[X] = np$ formula is standard.