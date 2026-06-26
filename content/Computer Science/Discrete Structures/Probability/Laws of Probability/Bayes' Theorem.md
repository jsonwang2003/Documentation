
> [!ABSTRACT]
> 
> Bayes' Theorem provides a mathematical framework for updating the probability of a hypothesis (an event) as more evidence or information becomes available. It essentially allows us to "reverse" conditional probabilities.

---
## The Formula
The theorem is derived from the definition of **[[Conditional Probabilities]]**. Since $P(E \cap F) = P(F|E)P(E)$ and $P(E \cap F) = P(E|F)P(F)$, we can equate them to solve for $P(F|E)$:

$$
P(F|E) = \frac{P(E|F)P(F)}{P(E)}
$$

Using the **Law of Total Probability**, we can expand the denominator $P(E)$ to account for all possible scenarios (where $F$ occurs and where $F$ does not occur, denoted as $\overline{F}$):

$$
\boxed{P(F|E) = \frac{P(E|F)P(F)}{P(E|F)P(F) + P(E|\overline{F})P(\overline{F})}}
$$

---
## Step-by-Step Example: Steroid Testing
**The Problem:**
- A test detects steroids **95% of the time** (True Positive).
- The test has a **15% false positive rate** for clean athletes.
- Only **10% of athletes** actually use steroids.
- **Question:** If an athlete tests positive, what is the probability they actually used steroids?
### 1. Identify the Variables
- $F$: The athlete **used steroids** (The hypothesis).
- $E$: The athlete **tested positive** (The evidence).
- $P(F) = 0.10$: Prior probability of steroid use.
- $P(\overline{F}) = 0.90$: Prior probability of being clean.
- $P(E|F) = 0.95$: Probability of testing positive _given_ steroid use.
- $P(E|\overline{F}) = 0.15$: Probability of testing positive _given_ no steroid use.

### 2. Apply the Theorem
Substitute the values into the expanded formula:

$$
\begin{align*}
P(F|E) &= \frac{P(E|F)P(F)}{P(E|F)P(F) + P(E|\overline{F})P(\overline{F})} \\
&= \frac{0.95 \cdot 0.1}{0.95 \cdot 0.1 + 0.15 \cdot 0.9} \\
&= \frac{0.095}{0.095 + 0.135} \\
&= \boxed{0.41} 
\end{align*}
$$

### 3. Visualizing the Result

Even though the test is "95% accurate" at detection, a positive result only means there is a **41% chance** the athlete is guilty. This occurs because the number of "clean" athletes is much larger than the number of "users," so the 15% false positive rate generates more total positive results than the 95% detection rate does.
#### Explanation
In the universal set, there are 2 possible associations
1. Did use steroid ($10\%$)
2. Did not use steroid ($90\%$)

![[Pasted image 20251018195619.png]]

When the company said "drug test will **detect steroid use $95\%$ of the time**," It was really only looking at those who did use steroid (orange $10\%$). At the same time, the drug falsely tested positive on those who did not use steroids $15\%$ of the time.

![[Pasted image 20251018200049.png]]

The resulting $41\%$ is the percentage of those who used steroids given that the drug test came out positive

![[Pasted image 20251018200431.png]]

---
## Common Use Cases

| **Field**            | **Application**                                                                    |
| -------------------- | ---------------------------------------------------------------------------------- |
| **Medical Testing**  | Determining the actual likelihood of a disease given a positive lab result.        |
| **Spam Filtering**   | Calculating the probability a message is spam given the presence of certain words. |
| **Machine Learning** | [[Naïve Bayes]] classifiers use these principles for categorization tasks.         |

---
## Related Notes
- **[[Conditional Probabilities]]**: The foundational logic for Bayes' Theorem.
- **[[Case Analysis]]**: Bayes' Theorem is often the final step after a case analysis of a multi-stage process.
- **[[Independent Events]]**: If $E$ and $F$ are independent, $P(F|E) = P(F)$, and the evidence provides no new information.