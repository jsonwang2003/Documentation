---
title: Solving Recurrence Closed Form
---
> [!ABSTRACT]
> 
> This toolkit contains the mathematical strategies required to transform recurrence relations—where a function is defined in terms of itself—into closed-form expressions. Converting to closed form allows for direct calculation of values and clear [[Asymptotic Notation|asymptotic analysis]].

---
## The Primary Solving Techniques

- **[[Master Theorem]]**
    - **Usage:** A rapid "plug-and-play" shortcut for divide-and-conquer recurrences of the form $T(n) = aT(n/b) + O(n^d)$.
    - **Logic:** Compare $a$ with $b^d$ to determine if the work is concentrated at the root, the leaves, or evenly distributed.
- **[[Unraveling]] (Iteration Method)**
    - **Usage:** Best for recurrences that are easy to expand step-by-step to find a summation pattern.
    - **Step:** Repeatedly substitute the recurrence into itself ($k$ times) until a pattern emerges, then solve the resulting summation.
- **[[Guess and Check]] (Substitution Method)**
    - **Usage:** When a pattern is easily spotted from small values of $n$.
    - **Constraint:** Requires a formal **Proof by Induction** to verify the guess is mathematically sound for all $n$.
- **[[Homogeneous Recurrence Relations with Constant Coefficients (HRRCC)|Characteristic Polynomial]]**
    - **Usage:** For linear recurrences like Fibonacci where $T(n)$ depends on multiple previous terms.
    - **Logic:** Solve the roots of the characteristic equation to find the growth constant $r$.

---
## Master Theorem Quick-Reference

For $T(n) = aT(n/b) + O(n^d)$, the complexity is determined by:

|**Condition**|**Result**|**Intuition**|
|---|---|---|
|**$a < b^d$**|$O(n^d)$|Work is dominated by the non-recursive part.|
|**$a = b^d$**|$O(n^d \log n)$|Work is distributed evenly across tree levels.|
|**$a > b^d$**|$O(n^{\log_b a})$|Work is dominated by the recursive calls (leaves).|

---
## 🧮 Summary Comparison

| **Technique**                                                                                                     | **Effort** | **Reliability** | **Best Use Case**                         |
| ----------------------------------------------------------------------------------------------------------------- | ---------- | --------------- | ----------------------------------------- |
| **Master Theorem**                                                                                                | 🟢 Low     | 🟡 High         | Standard $T(n/b)$ forms.                  |
| **Unraveling**                                                                                                    | 🟡 Med     | 🟢 High         | Single-term recurrences (Hanoi/Sums).     |
| **Guess & Check**                                                                                                 | 🟡 Med     | 🔴 Low*         | When the pattern is obvious (Power of 2). |
| **HRRCC**                                                                                                         | 🔴 High    | 🟢 High         | Linear sequences (Fibonacci).             |
> [!NOTE]
> Low reliability refers to the difficulty of making the initial guess, not the validity of the induction proof.

---
# Examples
### Pair of Elements
Find the number of pairs of elements from a set of size $n$

Let $P(n)$ be the number of unordered pairs of elements from a set of size $n$

#### Find the Pattern

| $n$ | $P(n)$ |
| --- | ------ |
| 1   | 0      |
| 2   | 1      |
| 3   | 3      |
| 4   | 6      |
| 5   | 10     |
We find the pattern to be matching that of the $2^{nd}$ diagonal column from [[Pascal's Identity|Pascal's Triangle]]

> [!NOTE]
> There is another definition for this sequence, it is called *Triangle Numbers*
> This is due to the number of elements that can arrange them into a equilateral triangle
> 
> ![[Pasted image 20251116224009.png]]
#### Partition the Set
Splitting all the pairs into 2 disjoint sets

$$
\begin{align*}
\text{number of pairs} &= \text{all pairs containing n}\\
&+ \text{all pairs that does not contain n}\\\\
P(n) &= P(n-1) + (n-1)\\
P(1) &= 0
\end{align*}
$$
![[Pasted image 20251116224546.png]]
#### Solution
##### Unraveling
$$
\begin{align*}
P(n) &= P(n-1) + (n-1)\\
P(n) &= [P(n-2) + (n-2)] + (n-1)\\
&= P(n-2) + (n-2) + (n-1)\\
P(n) &= [P(n-3) + (n-3)] (n-2) + (n-1)\\
&= P(n-3) + (n-3) + (n-2) + (n-1)\\
&\vdots\\
P(n) &= P(n-k) + \sum_{i = n-k}^{n-1} i\\
&\vdots\\
\text{let } k &= n-1 \text{to reach base case}\\
P(n) &= P(1) + \sum_{i = 1}^{n-1}\\
&= \sum_{i = 1}^{n-1} i\\
&= \boxed{\frac{n(n-1)}{2}}
\end{align*}
$$
##### Guess and Check
Guess $P(n) = \frac{n(n-1)}{2}$
- $P(1) = \frac{1(0)}{2} = 0$     $\checkmark$
- $P(2) = \frac{2(1)}{2} = 1$     $\checkmark$
- $P(3) = \frac{3(2)}{2} = 3$     $\checkmark$

Requires an induction proof to verify that the guess I have is correct
**Claim**: 
	$P(n) = \frac{n(n-1)}{2}$ for all $n \geq 1$
**Base Case**: 
	$P(1) = \boxed{0} = \frac{1(1-1)}{2}$ 
	Base case holds
**Inductive Hypothesis**: 
	Suppose that for some $k \geq 1$, $P(k) = \frac{k(k-1)}{2}$
**Inductive Step**: 
	Want to show that $P(k+1) = \frac{(k+1)(k)}{2}$
$$
\begin{align*}
P(k+1) &= P(k) + k \\
&= \frac{k(k-1)}{2} + k\\
&= \frac{k(k-1) + 2k}{2}\\
&= \frac{k^2-k+2k}{2}\\
&= \frac{k^2 + k}{2}\\
P(k+1) &= \boxed{\frac{k(k+1)}{2}}
\end{align*}
$$

Since Induction holds, the closed form for $P(n) = \frac{n(n-1)}{2}$

### The Tower of Hanoi
How many moves it take to relocate all disks to another pole?
- Can only move one disk at a time
- Cannot put a larger disk on top of a smaller disk

#### Find the Pattern
Let $T(n)$ be the number of moves to solve puzzle with $n$ disks

| $n$ | $T(n)$ |
| --- | ------ |
| 1   | 1      |
| 2   | 3      |
| 3   | 7      |
| 4   | 15     |
By the pattern, we can find the $T(n) = 2^n-1$, $T(1) = 1$

> [!NOTE]
> Might be helpful to think recursively in order to prove the correctness of $2^n-1$
> 
> Recursive solution
> 1. Move the the stack of the smallest $n-1$ disks to an empty pole $T(n-1)$
> 2. Move the largest disk to the remaining empty pole
> 3. Move the stack of the smallest $n-1$ disks to the pole with the largest disk
> 
> $$T(n) = T(n-1) + 1 + T(n-1) = \boxed{2T(n-1) + 1}$$

#### Solution
##### Guess and Check
Guess $T(n) = 2^n-1$
- $P(1) = 2^{(1)}-1 = 1$     $\checkmark$
- $P(2) = 2^{(2)} - 1= 3$     $\checkmark$
- $P(3) = 2^{(3)} - 1 = 7$     $\checkmark$

Requires Induction Proof
**Claim**: 
	For each positive integer $n$, $T(n) = 1$
**Base Case**: 
	if $n=1$, then $T(n) = 1$
	According to the recurrence, plugging $n=1$ into the formula gives $T(1)=2^1-1 = 2-1=1$ $\checkmark$
**Inducive Hypothesis**
	Suppose $n$ is a positive integer greater than $1$ and, as the induction hypothesis, assume that $T(n-1) = 2^{n-1}-1$.
**Inductive Step**
	We need to show that $T(n) = 2^n-1$
$$
\begin{align*}
T(n) &= 2T(n-1) + 1\\
&= 2(2^{n-1}-1)+1\\
&= 2^n-2+1\\
&= \boxed{2^n-1}
\end{align*}
$$
##### Unravel
$$
\begin{align*}
T(n) &= 2T(n-1) + 1\\
T(n) &= 2(2T(n-2)+1)+1\\
&= 2^2T(n-2)+2+1\\
T(n) &= 2^2(2T(n-3)+1)+2+1\\
&= 2^3T(n-3) +2^2+2+1\\
&\vdots\\
T(n) &= 2^kT(n-k) + 2^{k-1}+2^{k-2} + 2^2+2+1\\
&= 2^kT(n-k) + 2^k-1\\\\
\text{let } k &= n-1 \text{ to drop to base case}\\
T(n) &= 2^{n-1}T(1)+2^{n-1}-1\\
&= 2 \cdot2^{n-1} -1\\
&= 2^n-1
\end{align*}
$$