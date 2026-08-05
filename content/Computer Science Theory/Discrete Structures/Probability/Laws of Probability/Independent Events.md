> [!ABSTRACT]
> 
> Independence describes a relationship between two or more events where the occurrence (or non-occurrence) of one provides no information about the likelihood of the others. It is a fundamental property used to simplify complex probability calculations.

---
## Defining Independence
Two events $E$ and $F$ are **independent** if the occurrence of one does not change the probability of the other. Mathematically, this is expressed through **[[Conditional Probabilities]]**:

$$
P(E|F) = P(E)
$$

Using the definition of conditional probability, we derive the Product Rule for Independent Events:

$$
\boxed{P(E \cap F) = P(E)P(F)}
$$

If this equality holds, the events are independent. If it does not, the events are **dependent**.

---
## Independent vs. Disjoint Events
A common point of confusion is the difference between independent events and disjoint (mutually exclusive) events.

> [!IMPORTANT]
> 
> Independent events are NOT the same as disjoint events. In fact, if two events with non-zero probability are disjoint, they must be dependent because the occurrence of one guarantees the other did not occur ($P(A|B) = 0$).

|**Feature**|**Independent Events**|**Disjoint Events**|
|---|---|---|
|**Can occur together?**|Yes|No|
|**Influence**|No influence on each other|One excludes the other|
|**Formula**|$P(A \cap B) = P(A) \cdot P(B)$|$P(A \cap B) = 0$|
|**Visual**|Overlapping regions|Non-overlapping circles|

![[Pasted image 20251217215032.png]]

---
## Example: Bitstring Properties
Suppose we generate a random bitstring of length 4.
- **Event $E$**: The string starts with a **1**.
- **Event $F$**: The string contains an **even number of 1s**.

**Step 1: Calculate individual probabilities**
- $|S| = 2^4 = 16$
- $|E| = 8$ (half the strings start with 1), so $P(E) = \frac{8}{16} = \frac{1}{2}$.
- $|F| = 8$ (in bitstrings of length $n$, exactly half have an even number of 1s), so $P(F) = \frac{8}{16} = \frac{1}{2}$.

**Step 2: Calculate the intersection**
- $E \cap F = \{1001, 1010, 1100, 1111\}$
- $|E \cap F| = 4$, so $P(E \cap F) = \frac{4}{16} = \frac{1}{4}$.

**Step 3: Test for independence**
- Does $P(E \cap F) = P(E) \cdot P(F)$?
- $\frac{1}{4} = \frac{1}{2} \cdot \frac{1}{2}$

Since the equality holds, events $E$ and $F$ are **independent**. Knowing that the string starts with a 1 gives you absolutely no advantage in guessing if the total count of 1s is even or odd.

---
## Pairwise vs. Mutual Independence
When dealing with more than two events ($E_1, E_2, \dots, E_n$):
- **Pairwise Independence**: Every possible pair $(E_i, E_j)$ satisfies $P(E_i \cap E_j) = P(E_i)P(E_j)$.
- **Mutual Independence**: The probability of the intersection of _any_ subset of events is the product of their individual probabilities.

> [!NOTE]
> 
> Pairwise independence does not guarantee mutual independence.

---
## Related Notes
- **[[Conditional Probabilities]]**: The foundation for defining how events interact.
- **[[Linearity of Expectation]]**: A property that holds true regardless of whether events are independent or dependent.
- **[[Binomial Distribution]]**: Built on the assumption of $n$ independent Bernoulli trials.