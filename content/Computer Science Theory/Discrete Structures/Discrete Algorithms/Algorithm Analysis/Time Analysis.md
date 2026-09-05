> [!ABSTRACT]
> 
> In time analysis, we care about the growth rate of a function as $n$ approaches infinity. This allows us to ignore hardware differences and focus on the scalability of the algorithm itself.

---
## The Hierarchy of Functions
Ordered from **slowest growth** (most efficient) to **fastest growth** (least efficient):

|**Notation**|**Name**|**Typical Example**|
|---|---|---|
|$O(1)$|**Constant**|Accessing an array index|
|$O(\log(\log n))$|**Double Logarithm**|Advanced data structure operations|
|$O(\log n)$|**Logarithmic**|Binary Search|
|$O(\log^k n)$|**Poly-logarithmic**|Complex nested logs|
|$O(n)$|**Linear**|Linear Search, Single loop|
|$O(n \log n)$|**Log-linear**|Merge Sort, Quick Sort|
|$O(n^2)$|**Quadratic**|Bubble Sort, Nested loops|
|$O(n^k)$|**Polynomial**|Triple nested loops ($k=3$)|
|$O(a^n)$|**Exponential**|Recursive Fibonacci, Subset Sum|
|$O(n!)$|**Factorial**|Traveling Salesperson (Brute force)|
|$O(n^n)$|**Super-Exponential**|Extremely inefficient recursion|

---
## Comparison Rules
To determine which algorithm is "better" asymptotically, use these dominant growth rules:
1. **Polynomials vs. Logarithms**: Any positive power of $n$ ($n^{0.0001}$) eventually grows faster than any power of $\log n$.
2. **Polynomials vs. Exponentials**: Any exponential ($1.1^n$) eventually grows faster than any polynomial ($n^{100}$).
3. **The Base Matters**:
    - For $n^a$ vs $n^b$: The larger exponent wins ($n^3 > n^2$).
    - For $a^n$ vs $b^n$: The larger base wins ($3^n > 2^n$).
4. **Factorials are Massive**: $n!$ grows faster than $a^n$, but slower than $n^n$.

---
## Simplification Rules (Big-O Properties)
When analyzing code to find the final complexity, follow these three logic steps:
### 1. Drop the Constants
As $n \to \infty$, fixed multipliers become irrelevant.
- $5n^2 + 10n + 3 \implies O(n^2)$
- $O(2n) = O(n)$

### 2. Drop Non-Dominant Terms
Only the "fastest growing" term matters.
- $O(n^2 + n \log n + n) \implies O(n^2)$
- $O(n! + 2^n) \implies O(n!)$

### 3. Products and Sums
- **Sequential Steps**: If you do $O(f(n))$ work then $O(g(n))$ work, the total is $O(f(n) + g(n))$.
- **Nested Steps**: If you do $O(g(n))$ work _inside_ a loop that runs $f(n)$ times, the total is $O(f(n) \cdot g(n))$.

---
## Related Notes
- [[Asymptotic Notation]] — The formal definitions of $O$, $\Omega$, and $\Theta$.
- [[Time Analysis For Recursion]] — Applying these rules to $T(n)$ relations.
- [[Stirling's Approximation]] — How to handle the growth of $n!$.