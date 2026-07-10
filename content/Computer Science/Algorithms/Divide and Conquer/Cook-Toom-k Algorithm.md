---
description: Generalizes Karatsuba's multiplication trick to a k-way split, combining subproblems with 2k-1 multiplications instead of k^2, approaching O(n^(1+ε)) as k grows.
tags:
  - algorithm
  - divide-and-conquer
aliases:
  - Toom-Cook
  - Cook-Toom
  - Multiply KS
---

> [!abstract] Abstract 
> Recall the [[Computer Science/Algorithms/Divide and Conquer/index#Multiplying n-bit Numbers (Classic Recursive Approach)|Multiply Problem]]: splitting two $n$-bit numbers into halves and recursively multiplying the pieces. Cook-Toom-$k$ asks: what happens if we divide into $k$ subproblems, each of size $\frac{n}{k}$, instead of just 2?
> 
> - **Category:** Divide and Conquer / Integer Multiplication
> - **Input:** Two $n$-bit numbers (represented as degree-$(k-1)$ polynomials once split into $k$ parts)
> - **Output:** The product
> - **Paradigm:** Divide and Conquer, generalizing [[Computer Science/Algorithms/Divide and Conquer/index#Multiplying n-bit Numbers (Classic Recursive Approach)|Karatsuba's]] 2-way split to a $k$-way split
> - **Typical use cases:** fast big-integer / big-polynomial multiplication; a stepping stone toward understanding how multiplication can approach near-linear time

---

# Core Logic (High-Level)

## The Naive k-Way Split

Splitting each number into $k$ equal parts represents it as a degree-$(k-1)$ polynomial:

$$ (a_{k-1}x^{k-1} + a_{k-2}x^{k-2} + \dots + a_{1}x + a_{0})(b_{k-1}x^{k-1} + b_{k-2}x^{k-2} + \dots + b_{1}x + b_{0}) $$

Multiplying these two polynomials the schoolbook way requires $k^2$ coefficient multiplications (every $a_i$ against every $b_j$). The recursion:

$$ 
\begin{align*} 
T(n) &= k^{2} T\left( \frac{n}{k} \right) + O(n) \qquad (a = k^{2},\ b=k,\ d=1)\\
&= O(n^{\log_{k}k^{2}})\\
&= \boxed{O(n^{2})} 
\end{align*} 
$$

So a naive $k$-way split, at _any_ $k$, is still $O(n^2)$ — exactly as slow as the classic 2-way split. Splitting into more pieces doesn't help unless the _combine_ step also improves.

> [!tip] Key Idea 
> If you split a number into $k$ equally-sized parts, you can combine them with only $\boxed{2k-1}$ multiplications instead of $k^2$ — by evaluating both degree-$(k-1)$ polynomials at $2k-1$ distinct points, multiplying the resulting values pointwise, and interpolating to recover the product polynomial's coefficients. (This evaluate → pointwise-multiply → interpolate structure is the general Toom-Cook technique; Karatsuba is the special case $k=2$, needing only $2(2)-1 = 3$ multiplications — matching the "Multiply KS" trick in [[Computer Science/Algorithms/Divide and Conquer/index#Multiplying n-bit Numbers (Classic Recursive Approach)|Multiply Problem]].)

---

# Pseudocode (Mid-Level Implementation)

```pseudo
	\begin{algorithm}
	\caption{Cook-Toom-k Multiply}
	\begin{algorithmic}
	\Input Two $n$-bit numbers $x, y$; a chosen split factor $k$
	\Output the product $xy$
	\Procedure{CookToomK}{$x, y, k$}
		\If{$n$ is small enough}
			\Return $xy$ directly
        \EndIf
        \State Split $x, y$ into $k$ parts each, forming coefficients $a_0,\dots,a_{k-1}$ and $b_0,\dots,b_{k-1}$
        \State Choose $2k-1$ distinct evaluation points $z_1, \dots, z_{2k-1}$
        \For{each point $z_i$}
	        \State Evaluate $A(z_i) = \sum_j a_j z_i^j$ and $B(z_i) = \sum_j b_j z_i^j$
	        \State $R_i$ = CookToomK($A(z_i)$, $B(z_i)$, $k$) \Comment{recursive call, size $n/k$}
        \EndFor
        \State Interpolate the product polynomial's coefficients from the $2k-1$ values $R_1, \dots, R_{2k-1}$
        \Return the recombined product $xy$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`k`|Integer (fixed, chosen ahead of time)|Number of equal parts each number is split into|
|`a_0,...,a_{k-1}` / `b_0,...,b_{k-1}`|Coefficients|The $k$-part representation of $x$ and $y$ as degree-$(k-1)$ polynomials|
|`z_1,...,z_{2k-1}`|Evaluation points|Chosen points used to reduce polynomial multiplication to $2k-1$ pointwise scalar multiplications|

## Helper Functions / Operations Used

- **Evaluate** — computing $A(z_i)$, $B(z_i)$ for each point; $O(k)$ per point, $O(k^2)$ total across all $2k-1$ points.
- **Interpolate** — solving for the product polynomial's coefficients given its values at $2k-1$ points (e.g. via Lagrange interpolation); also $O(k^2)$-ish, contributing to the combine-step overhead $M(k)$ below.

---

# Proof of Correctness

The correctness argument is essentially interpolation theory rather than an algorithmic invariant: a polynomial of degree $\leq 2k-2$ (the true product of two degree-$(k-1)$ polynomials) is uniquely determined by its values at any $2k-1$ distinct points. So evaluating both inputs at $2k-1$ points, multiplying pointwise (which correctly gives the product polynomial's value at each of those points), and interpolating recovers the exact product — no approximation is involved, provided the evaluation points are distinct.

---

# Time & Space Complexity Analysis

## General Case

With the improved $2k-1$-multiplication combine step:

$$ 
\begin{align*} 
T(n) &= (2k-1)T\left( \frac{n}{k} \right) + M(k)\cdot n\\
&= O\left(n^{\frac{\log(2k-1)}{\log k}}\right) 
\end{align*} 
$$

where $M(k) = O(k^2)$ is the coefficient of the linear combine-step term (evaluation + interpolation overhead) — note this is still _linear in $n$_ since $k$ is fixed, but the constant factor in front grows quadratically with $k$.

## Implementation-Dependent Variations

| Choice of $k$     | Exponent $\frac{\log(2k-1)}{\log k}$ | Combine overhead $M(k)$               | Notes                                                                                                                                                            |
| ----------------- | ------------------------------------ | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $k=2$ (Karatsuba) | $\log_2 3 \approx 1.585$             | Small, constant                       | The classic special case — see [[Computer Science/Algorithms/Divide and Conquer/index#Multiplying n-bit Numbers (Classic Recursive Approach)\|Multiply Problem]] |
| Larger $k$        | Approaches $1$ as $k \to \infty$     | Grows as $O(k^2)$ — larger and larger | Better asymptotic exponent, but a rapidly growing constant factor ― practically not worth it cause the **non-recursive** part grows *quadratically*              |

## Best / Worst / Average Case

- **Best / Worst / Average case:** All the same order for a fixed $k$ — the algorithm always performs the same evaluate/recurse/interpolate structure regardless of the specific input values, so there's no input-dependent variation, only $k$-dependent variation.

---

# Drawbacks / Constraints

- **The constant factor grows fast with $k$.** $M(k) = O(k^2)$ means choosing a larger $k$ to shrink the exponent isn't free — the linear term's coefficient grows quadratically, so there's a real trade-off, not a free lunch.
- **Only approaches linear time, never reaches it.** Since $\lim_{k\to\infty} \frac{\log(2k-1)}{\log k} = 1$, for any $\epsilon > 0$ you can choose $k$ large enough to get $O(n^{1+\epsilon})$ — but no fixed choice of $k$ actually reaches $O(n)$ or even $O(n\log n)$. Achieving that requires a fundamentally different approach (FFT-based multiplication, e.g. Schönhage–Strassen).
- **Numerical/implementation care needed:** evaluation points must be chosen so interpolation is well-conditioned (e.g. small integers), since poorly chosen points can blow up coefficient sizes or cause numerical instability in the interpolation step.

---

# References / Links

- [[Computer Science/Algorithms/Divide and Conquer/index#Multiplying n-bit Numbers (Classic Recursive Approach)|Multiply Problem]]
- [[Master Theorem]]
- [[Computer Science/Algorithms/Divide and Conquer/index|Divide and Conquer]]