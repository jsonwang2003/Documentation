
> [!ABSTRACT]
> 
> Conditional probability measures the likelihood of an event occurring given that another event has already occurred. This effectively restricts the sample space to only those outcomes where the second event is true, rescaling the probabilities accordingly.

---
## The Formula
Suppose $A$ and $B$ are events, and $P(B) > 0$. The probability of $A$ given $B$ is defined as:

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

- **$P(A|B)$**: The probability of $A$ given $B$.
- **$P(A \cap B)$**: The probability of the **intersection** (both $A$ and $B$ occurring).
- **$P(B)$**: The probability of $B$, which serves to **rescale the scope** of the probability to the new restricted sample space.

From this, we can also derive the Multiplication Rule:

$$
P(A \cap B) = P(A|B)P(B)
$$

![[Pasted image 20251018190501.png]]

---
## Example: Two Dice
Consider rolling a fair blue die and a fair yellow die. What is the probability that **the sum is 8** ($A$) given that **both dice are even** ($B$)?
### 1. Identify the Events
- **Event $B$ (Condition)**: Both dice are even.
    - Possible outcomes: $(2,2), (2,4), (2,6), (4,2), (4,4), (4,6), (6,2), (6,4), (6,6)$
    ![[Pasted image 20251018190818.png]]
    
    - $P(B) = \frac{9}{36}$.
- **Event $A \cap B$ (Intersection)**: The sum is 8 **and** both dice are even.
    - Possible outcomes: $(2,6), (4,4), (6,2)$.
    
	![[Pasted image 20251018191159.png]]
    
    - $P(A \cap B) = \frac{3}{36}$.

### 2. Calculate the Conditional Probability
Applying the formula:

$$
P(A|B) = \frac{\frac{3}{36}}{\frac{9}{36}} = \frac{3}{9} = \boxed{\frac{1}{3}}
$$

---
## Key Intuition

The probability of an event **may change** if you have additional information about the outcomes. In the example above, the unconditional probability of the sum being 8 is $\frac{5}{36}$. However, knowing both dice are even increases that probability to $\frac{1}{3}$ (or $\frac{12}{36}$) because we have eliminated all "odd-sum" and "mixed-parity" possibilities that could not result in our specific even-sum condition.

---
## Related Notes
- **[[Independent Events]]**: Events where $P(A|B) = P(A)$, meaning the additional information does not change the likelihood.
- **[[Bayes' Theorem]]**: A method for "reversing" conditional probabilities to find $P(B|A)$ if you know $P(A|B)$.
- **[[Distributions#Law of Total Probability|Law of Total Probability]]**: Used to calculate the total probability of an event by summing its conditional probabilities across a partition.