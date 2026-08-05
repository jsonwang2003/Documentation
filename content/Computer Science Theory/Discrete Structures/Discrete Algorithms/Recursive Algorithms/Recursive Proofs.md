> [!ABSTRACT]
> 
> A Recursive Proof is a formal verification method used to guarantee that a recursive algorithm produces the correct output for all valid inputs. It relies on the mathematical symmetry between recursion and induction: the base cases of the recursion serve as the base cases of the induction.

---
## The Big Picture: Induction vs. Recursion
While they look similar, they serve different purposes in computer science:
- **Recursion** is a **computational strategy** for solving a problem by breaking it into smaller instances.1
- **[[Induction]]** is a **logical strategy** for proving a statement is true for an infinite set of cases.

To prove a recursive algorithm is correct, we use **Strong Induction** on the size of the input $n$.

---
## The Formal Proof Structure
When writing a proof for a recursive algorithm, follow these four distinct steps:

**Example Code:**
```python
countDoubleRec(string, n):
	if n < 2:
		// Base Case
		return 0
	if string[0,1] == "00":
		// Recursive Call A
		return 1 + countDoubleRec(string[1:], n-1)
	else:
		// Recursive Call B
		return countDoubleRec(string[1:], n-1)
```
### 1. The Claim
State clearly what the algorithm is supposed to do.
- _Example:_ "Claim: `countDoubleRec(S)` returns the total number of occurrences of the substring '00' in string $S$ for all lengths $n \geq 0$."

### 2. Base Case(s)
Identify the smallest possible inputs where the recursion stops.
- **Action:** State what the algorithm does for these inputs and explain **why** that is the correct answer.
- _Note:_ If the algorithm calls $T(n-2)$, you likely need two base cases ($n=0$ and $n=1$).

### 3. Inductive Hypothesis (Strong)
Assume the algorithm works correctly for all "smaller" inputs.
- **Formulation:** "Assume that for some $k \geq \text{base case}$, the algorithm is correct for all inputs of size $i$, where $\text{base case} \le i \le k$."
- **Why Strong?** Recursive calls don't always go to $n-1$. In **Divide and Conquer**, the call might be to $n/2$. Strong induction covers all possible smaller sizes.

### 4. Inductive Step (The Goal)
Show that if the algorithm is correct for size $k$, it **must** be correct for size $k+1$.
1. **Trace the Logic:** Look at the recursive case in the code.
2. **Apply Hypothesis:** Replace the recursive calls with "the correct answer according to the problem" (since our hypothesis says the calls work).
3. **Algebraic/Logical Wrap-up:** Show that the current step + the result of the recursive calls equals the expected total for size $k+1$.

---
## Applied Example: `FindMax` Proof

**Algorithm:** `FindMax(A, n)` returns $A[1]$ if $n=1$, otherwise it returns $MAX(FindMax(A, n-1), A[n])$.
- **Base Case ($n=1$):** The algorithm returns $A[1]$. Since a list of length 1 only has one element, that element is the maximum. **Correct.**
- **Inductive Hypothesis:** Assume `FindMax` correctly returns the maximum of any array of size $k$.
- **Inductive Step ($k+1$):**
    1. The algorithm computes $M_1 = FindMax(A, k)$ and returns $MAX(M_1, A[k+1])$.
    2. By our **Hypothesis**, $M_1$ is the true maximum of the first $k$ elements.
    3. The maximum of the first $k+1$ elements must be either the maximum of the first $k$ elements OR the new $(k+1)^{th}$ element.
    4. Since the algorithm compares these two specific values and returns the larger one, it correctly finds the maximum for $k+1$. → **Verified.**

---
## Regular vs. Strong Induction in Proofs

|**Feature**|**Regular Induction**|**Strong Induction**|
|---|---|---|
|**Logic**|Uses $(n-1)$ to prove $n$.|Uses **any/all** values $< n$ to prove $n$.|
|**Use Case**|Linear recursion ($n-1$).|Divide and Conquer ($n/2$, $n/3$) or HRRCC.|
|**Stability**|Easier to write but limited.|Required for almost all non-trivial algorithms.|