---
title: Identities
---
# Combinatorial Identities & Laws

> [!NOTE]
> 
> Identities are mathematical truths that remain equal for all values. In combinatorics, these often represent two different ways of counting the same set, proving that the left-hand side and right-hand side are logically equivalent.

---
## The Fundamental Laws
_Logic-based identities used to simplify the boundaries of sets._
- **[[Demorgan's Law]]**
    - **The Law:** $\neg(A \cup B) = \neg A \cap \neg B$ and $\neg(A \cap B) = \neg A \cup \neg B$.
    - **Why it matters:** Essential for switching between "OR" problems and "AND" problems, especially when using the Complement Rule.
- **[[Sum Identity]]**
    - **The Law:** The total number of ways to choose any number of elements from a set of $n$ is $2^n$.
    - **Connection:** $\sum_{k=0}^{n} \binom{n}{k} = 2^n$.

---
## Binomial & Pascal Identities
_The DNA of the Pascal Triangle and polynomial expansions._
- **[[Binomial Theorem]]**
    - **The Law:** $(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k$.
    - **Why it matters:** Links algebra to combinatorics; the coefficients of expanded polynomials are exactly the combinations $C(n, k)$.
- **[[Pascal's Identity]]**
    - **The Law:** $\binom{n+1}{k} = \binom{n}{k-1} + \binom{n}{k}$.
    - **Why it matters:** The recursive definition of the Pascal Triangle. It represents the choice of either including or excluding a specific element from a selection.

---
## Symmetry & Selection
_Identities that exploit the "mirror" nature of combinations._
- **[[Symmetry Identity]]**
    - **The Law:** $\binom{n}{k} = \binom{n}{n-k}$.
    - **Why it matters:** Choosing $k$ items to **keep** is mathematically identical to choosing $n-k$ items to **discard**. This significantly simplifies calculations when $k$ is large.

---
## Identity Quick-Reference

|**Identity Name**|**Mathematical Form**|**Conceptual Shortcut**|
|---|---|---|
|**Symmetry**|$C(n, r) = C(n, n-r)$|Keep vs. Discard|
|**Pascal's**|$\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$|Include vs. Exclude|
|**Binomial**|$(a+b)^n = \dots$|Polynomial Coefficients|
|**Null/Full**|$\binom{n}{0} = \binom{n}{n} = 1$|Only one way to take none or all|
