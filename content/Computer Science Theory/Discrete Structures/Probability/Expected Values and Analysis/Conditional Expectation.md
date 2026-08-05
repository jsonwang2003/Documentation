> [!ABSTRACT]
> 
> Conditional expectation is the expected value of a random variable given that a specific event has occurred. It allows for the calculation of average outcomes within a restricted subset of the sample space.

---
## Formula and Definition
The conditional expectation of a random variable $X$ given an event $A$ is defined as the weighted average of the values of $X$ for all outcomes in $A$:

$$
E[X|A] = \frac{1}{P(A)}\sum_{a \in A} P(a)X(a)
$$

This formula scales the probabilities of the outcomes within $A$ so that they sum to $1$ relative to the event $A$ itself.

---
## Law of Total Expectation
The **Law of Total Expectation** (also known as the Law of Iterated Expectations) allows you to calculate the global expectation of a random variable by partitioning the sample space into disjoint cases.

For any event $B$ and its complement $\overline{B}$:

$$
E(X) = E(X|B)P(B) + E(X|\overline{B})P(\overline{B})
$$

This is a powerful tool for solving problems where the outcome depends on an initial random condition or "stage".

---
## Examples
### 1. Dice Sum with a Constraint
Calculate the expected sum $X$ of two fair dice, given that the first die is greater than 4 (Event $A$).
- **Step 1**: Identify $P(A)$. There are 12 outcomes where the first die is 5 or 6, so $P(A) = 12/36$.
- **Step 2**: Sum the values of $X$ for all outcomes in $A$.
- **Step 3**: Apply the formula:
    $$
    \begin{align*} 
    E(X|A) &= \frac{1}{12/36} \left[ \sum_{a \in A} \frac{1}{36}X(a) \right] \\
    &= \frac{1}{12}(6+7+8+9+10+11 + 7+8+9+10+11+12) \\
    &= \frac{108}{12} = \boxed{9} 
    \end{align*}
    $$
    
### 2. Manufacturing Quality Control
A lightbulb has a $70\%$ chance of coming from Factory A ($E[X|A] = 5000$ hours) and a $30\%$ chance of coming from Factory B ($E[X|B] = 6000$ hours).

Using the Law of Total Expectation:

$$
\begin{align*}
E(X) &= E(X|A)P(A) + E(X|B)P(B) \\
&= (5000 \cdot 0.7) + (6000 \cdot 0.3) \\
&= 3500 + 1800 = \boxed{5300 \text{ hours}} 
\end{align*}
$$

---
## Related Notes
- **[[Case Analysis]]**: The practical application of the Law of Total Expectation to break down complex probability problems.
- **[[Expected Value]]**: The foundational concept of the "long-run average".
- **[[Conditional Probabilities]]**: The underlying rules governing how probabilities change given new information.