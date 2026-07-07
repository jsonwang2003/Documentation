> [!ABSTRACT]
> 
> Mathematical Induction is a proof technique used to prove that a statement $P(n)$ is true for all natural numbers $n \geq n_0$. It works like a chain of dominoes: if you can knock down the first one, and each falling domino knocks down the next, they all must eventually fall.

---
## 1. The Induction Procedure
To perform an induction proof, you must complete four distinct stages:
### I. Base Case
Show that the statement holds for the very first value in the set (usually $n = 0$ or $n = 1$).
- **Goal**: Verify $P(n_0)$ is true.
- **Purpose**: This provides the "starting point" for your proof.
### II. Inductive Hypothesis
Assume that the statement is true for some arbitrary integer $k$.
- **Goal**: State "Assume $P(k)$ is true for some $k \geq n_0$."
- **Note**: You are not claiming it is true for _all_ numbers yet, just picking a "random" step in the ladder.
### III. Inductive Step
Show that **if** $P(k)$ is true, then $P(k+1)$ **must** also be true.
- **Goal**: Prove the implication $P(k) \implies P(k+1)$.
- **Strategy**: Use the algebraic expression from your hypothesis to simplify the expression for $k+1$. This is the "engine" of the proof.
### IV. Conclusion
State that since the base case and the inductive step are both verified, the statement holds for all $n$ in the defined domain.
- **Formal Phrasing**: "By the Principle of Mathematical Induction, $P(n)$ is true for all $n \geq n_0$."

---
## 2. Example: Sum of Integers

**Claim**: $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$ for all $n \geq 1$.
1. Base Case ($n=1$):
LHS: $1$
RHS: $\frac{1(1+1)}{2} = \frac{2}{2} = 1$
LHS = RHS. The base case holds.
2. Inductive Hypothesis:
Assume that for some $k \geq 1$, the claim holds: 
$$
1 + 2 + \dots + k = \frac{k(k+1)}{2}
$$
3. Inductive Step:
We want to show $P(k+1)$ is true, i.e., 

$$
1 + 2 + \dots + k + (k+1) = \frac{(k+1)((k+1)+1)}{2}
$$

- Start with the LHS of $P(k+1)$:
    $$
    (1 + 2 + \dots + k) + (k+1)
    $$
    
- Substitute the Inductive Hypothesis:
    $$
    \frac{k(k+1)}{2} + (k+1)
    $$
    
- Factor out $(k+1)$:
    $$
    (k+1) \left( \frac{k}{2} + 1 \right) = (k+1) \left( \frac{k+2}{2} \right) = \frac{(k+1)(k+2)}{2}
    $$
    
    This matches the RHS of $P(k+1)$.

4. Conclusion:

Since the base case and inductive step are true, the claim is proven for all $n \geq 1$.

---
## 3. Strong Induction
In some cases, assuming only the _previous_ step ($k$) isn't enough to prove the next step ($k+1$). In **Strong Induction**, you assume the statement is true for **all** values from $n_0$ up to $k$.
- **Hypothesis**: Assume $P(n_0), P(n_0+1), \dots, P(k)$ are all true.
- **Application**: Frequently used in [[Time Analysis]] (like [[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Divide and Conquer/Merge Sort]]) or proving properties of prime numbers.

---
## 4. Induction in Algorithms
Induction is the primary method for proving:
- **[[Loop Invariants]]**: Proving that a property remains true through every iteration of a loop.
- **Recursive Correctness**: Proving that if a recursive call on a smaller input works, the current call works.