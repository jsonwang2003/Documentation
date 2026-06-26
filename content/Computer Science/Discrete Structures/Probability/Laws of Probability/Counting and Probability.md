> [!ABSTRACT]
> 
> Counting and Probability forms the foundation of discrete probability. It involves using combinatorial techniques to determine the size of events and sample spaces, allowing for the calculation of likelihoods when outcomes are equally probable.

---
## Uniform Probability Formula

> [!INFO]
> 
> If a sample space $S$ has a [[Uniform vs Non-Uniform Distributions#Uniform Distribution|Uniform Distribution]], then for any event $E \subseteq S$, the probability of $E$ is the ratio of the number of successful outcomes to the total number of possible outcomes:
> 
> $$
p(E) = \frac{|E|}{|S|}
$$

This formula is valid only when every individual outcome in the sample space $S$ has the same probability, $p(a) = \frac{1}{|S|}$.

---
## Combinatorial Examples
### 1. Simple Die Roll
The probability that a fair 6-sided die results in an **even number**:
- $S = \{1, 2, 3, 4, 5, 6\} \implies |S| = 6$
- $E = \{2, 4, 6\} \implies |E| = 3$
- $p(E) = \frac{3}{6} = \boxed{\frac{1}{2}}$
### 2. Strings with No Consecutive Identical Digits
What is the probability that a 4-digit string selected uniformly at random has **no two consecutive digits the same**?
- Calculate $|E|$: There are 10 choices for the first digit, and 9 choices for each subsequent digit (any digit except the one immediately preceding it).
    $$
    |E| = 10 \times 9 \times 9 \times 9 = 7,290
    $$
	
- Calculate $|S|$: Total possible 4-digit strings ($0000$ to $9999$).
    $$
    |S| = 10^4 = 10,000
    $$
    
- **Result**: $p(E) = \frac{7290}{10000} = \boxed{0.729}$
### 3. Digits in Increasing Order
What is the probability that a 4-digit string consists of **4 different digits in strictly increasing order**?
- Calculate $|E|$: Since the order is strictly increasing, any set of 4 unique digits can only be arranged in exactly one valid way. Therefore, we simply need to choose 4 digits out of 10.
    
    $$
    |E| = \binom{10}{4} = \frac{10!}{4!6!} = 210
    $$
    
- Calculate $|S|$: Total 4-digit strings.
    
    $$
    |S| = 10,000
    $$
    
- **Result**: $p(E) = \frac{210}{10000} = \boxed{0.021}$

---
## Key Principles

| **Principle**       | **Description**                                                                                       | **Formula**                       |
| ------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------- |
| **Sum Rule**        | Used when an event can occur in multiple disjoint (mutually exclusive) ways.                          | $$P(A \cup B) = P(A) + P(B)$$     |
| **Product Rule**    | Used for multi-stage processes where choices are independent.                                         | $$P(A \cap B) = P(A) \cdot P(B)$$ |
| **Complement Rule** | $p(E) = 1 - p(\overline{E})$. Often easier to calculate the probability of the event _not_ happening. | $$P(E) = 1 - P(\overline{E})$$    |

---
## Related Toolkits
- **[[Computer Science/Discrete Structures/Counting/index|Counting]]**: Advanced techniques like Permutations and Combinations used to find $|E|$ and $|S|$.
- **[[Uniform vs Non-Uniform Distributions]]**: Understanding when the $\frac{|E|}{|S|}$ formula is applicable.
- **[[Independent Events]]**: How the probability of a string property changes if the digits were not chosen independently.