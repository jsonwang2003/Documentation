> [!ABSTRACT]
> 
> Standard integer multiplication (the grade-school method) operates in $O(n^2)$ time. By utilizing the Divide and Conquer paradigm and clever algebraic identities, Karatsuba's Algorithm reduces the number of recursive multiplications required, breaking the quadratic barrier.
## Grade-School Multiplication ($O(n^2)$)
The traditional method relies on computing partial products for each digit and then summing them.
- **Mechanism**: Each of the $n$ digits of the first number is multiplied by each of the $n$ digits of the second number.
- **Cost**: This results in $n^2$ single-digit multiplications and $O(n^2)$ work in additions (accounting for carries).

---
## The Divide and Conquer Approach

To multiply two $n$-digit numbers $x$ and $y$, we split them into their high-order ($L$) and low-order ($R$) halves:
$$
	\begin{align*}
	x &= 10^{n/2}x_L + x_R\\
	y &= 10^{n/2}y_L + y_R
	\end{align*}
$$

The product $xy$ is expanded as:

$$
xy = 10^n(x_Ly_L) + 10^{n/2}(x_Ly_R + x_Ry_L) + x_Ry_R
$$

### 1. Naive Divide and Conquer
If we compute the four products ($x_Ly_L, x_Ly_R, x_Ry_L, x_Ry_R$) directly:
- **Recurrence**: $T(n) = 4T(n/2) + O(n)$
- **Analysis**: By Master Theorem ($a=4, b=2, d=1$), since $4 > 2^1$, the runtime is $O(n^{\log_2 4}) = \mathbf{O(n^2)}$.
- **Verdict**: No asymptotic improvement over the grade-school method.

---
## Karatsuba's Algorithm ($O(n^{1.585})$)

Anatolii Karatsuba discovered that we don't need all four products separately. We only need the **sum** of the middle terms $(x_Ly_R + x_Ry_L)$.

### The Algebraic Trick
Instead of four multiplications, we perform **three**:
1. $P_1 = x_L \cdot y_L$
2. $P_2 = x_R \cdot y_R$
3. $P_3 = (x_L + x_R) \cdot (y_L + y_R)$

The middle term is then derived via subtraction:
$$
x_Ly_R + x_Ry_L = P_3 - P_1 - P_2
$$

### Recurrence Runtime
Because we reduced the number of recursive calls from $4$ to $3$:

$$
T(n) = 3T(n/2) + O(n)
$$

**Using [[Master Theorem]]:**
- $a = 3$
- $b = 2$
- $d = 1$
Since $3 > 2^1$ (Case 3), the complexity is $O(n^{\log_b a})$.
→ **Result**: $O(n^{\log_2 3}) \approx \mathbf{O(n^{1.585})}$.

---
## Comparison of Methods

|**Method**|**Recurrence**|**Complexity**|**Efficiency**|
|---|---|---|---|
|**Grade-School**|N/A|$O(n^2)$|Baseline|
|**Naive D&C**|$4T(n/2) + O(n)$|$O(n^2)$|No gain|
|**Karatsuba**|$3T(n/2) + O(n)$|$O(n^{1.585})$|**Significantly Faster**|

---
## Related Notes
- [[Computer Science Theory/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Divide and Conquer/Merge Sort|Merge Sort]] — Another $O(n \log n)$ D&C application.
- [[Master Theorem]] — The tool used to analyze these runtimes.