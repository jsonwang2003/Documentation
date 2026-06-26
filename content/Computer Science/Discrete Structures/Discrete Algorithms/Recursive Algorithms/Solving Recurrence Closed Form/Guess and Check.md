> [!ABSTRACT]
> 
> The Guess and Check method is a two-phase strategy for solving recurrences. First, you use intuition or small-value testing to "guess" the closed form. Second, you use Mathematical Induction to "check" (prove) that your guess is correct for all $n$.

---
## The Two-Step Workflow

### Step 1: The Guess
Analyze the recurrence to find a pattern. You can do this by:
- **Tabulating Values**: Calculate $T(n)$ for $n = 1, 2, 3, 4, 5 \dots$ and look for familiar sequences (powers of 2, squares, factorials).
- **Unraveling**: Expand the recurrence a few times to see the general shape of the growth.

### Step 2: The Check (Induction)
Once you have a "Claim" ($T(n) = \text{your guess}$), you must prove it:
1. **Base Case**: Prove the claim holds for the smallest possible $n$ (usually $n=0$ or $n=1$).
2. **Inductive Hypothesis**: Assume the claim is true for some $k$ (i.e., $T(k) = \text{guess}$).
3. **Inductive Step**: Use the recurrence formula and your hypothesis to show that $T(k+1)$ also matches the guess.

---
## Case Studies
### 1. Pair of Elements (Triangle Numbers)
**The Recurrence:** $P(n) = P(n-1) + (n-1)$ with $P(1) = 0$.
- **The Guess:** By tabulating values $(0, 1, 3, 6, 10)$, we identify the pattern of **Triangle Numbers**. We guess $P(n) = \frac{n(n-1)}{2}$.
- The Check (Inductive Step):
    $$P(k+1) = P(k) + k$$
    Substitute the hypothesis: $\frac{k(k-1)}{2} + k = \frac{k^2-k+2k}{2} = \frac{k(k+1)}{2}$.
- **Result:** The guess is verified.
### 2. The Tower of Hanoi
**The Recurrence:** $T(n) = 2T(n-1) + 1$ with $T(1) = 1$.
- **The Guess:** Tabulating values $(1, 3, 7, 15)$ suggests the form $2^n - 1$.
- The Check (Inductive Step):
    $$T(k) = 2T(k-1) + 1$$
    
    Substitute the hypothesis: $2(2^{k-1}-1) + 1 = 2^k - 2 + 1 = 2^k - 1$.
- **Result:** The guess is verified.

---
## Pros and Cons

|**Feature**|**Description**|
|---|---|
|**Strength**|Extremely powerful for non-standard recurrences that don't fit the Master Theorem or HRRCC.|
|**Weakness**|You must guess the _exact_ form. If your guess is off by a constant (e.g., guessing $2^n$ instead of $2^n-1$), the induction will fail.|
|**Flexibility**|Can be used to prove **Upper Bounds** ($T(n) \leq cn^2$) even if you don't know the exact closed form.|

---
## Expert Tips
- **Loose Guesses**: If you can't find the exact closed form, try to prove an inequality (e.g., "I guess $T(n) \leq cn^2$") to establish the [[Asymptotic Notation#3. Big-O ($O$)|Big-O]] bound.
- **Inductive Strengthening**: Sometimes, the induction only works if you subtract a lower-order term from your guess (e.g., guessing $cn - d$ instead of just $cn$).