
> [!ABSTRACT] 
> HRRCC is a technique for finding the closed-form solution of linear recurrences where the current term is a linear combination of previous terms. It is the primary tool for analyzing sequences like Fibonacci and constraints on bit-strings.

---
## When To Use
A recurrence is an HRRCC if it matches the form:

$$f(n) = c_1f(n-1) + c_2f(n-2) + \dots + c_k f(n-k)$$

**Two Mandatory Conditions:**
1. **Homogeneity**: Every term on the right-hand side is a function of a previous $f(k)$. There are no "extra" constants or $n$ terms (e.g., no $+7$ or $+n$).
2. **Constant Coefficients**: The $c_i$ values must be constants, not functions of $n$.
## The "Characteristic" Method
1. **The Guess**: Assume the solution takes the form $f(n) = Ar^n$.
2. **The Polynomial**: Substitute the guess to create the **Characteristic Equation**:
    - For $f(n) = c_1f(n-1) + c_2f(n-2)$, the equation is $r^2 - c_1r - c_2 = 0$.
3. **Solve for Roots**: Find the roots ($r_1, r_2, \dots$) of the polynomial.
4. **General Form**:
    - **Distinct Roots**: $f(n) = A(r_1)^n + B(r_2)^n$.
    - **Multiplicity (Repeated Roots)**: If $r_1$ repeats twice, the form is $f(n) = A(r_1)^n + B \mathbf{n}(r_1)^n$.
5. **Solve for Constants**: Use **Base Cases** and Gaussian Elimination (or simple substitution) to find $A$ and $B$.

---
## Examples
### n-bit strings
An $n$-bit string is a string of length $n$ consisting of 0s and 1s

> [!NOTE]
> By [[Power Rule]] we already know the number of $n$-bit strings are $2^n$
> Try to find another way to find this value

#### Finding Pattern
Let $BS(n)$ be the number of n-bit strings

| $n$ | $BS(n)$ |
| --- | ------- |
| 0   | 1       |
| 1   | 2       |
| 2   | 4       |
| 3   | 8       |
| 4   | 16      |

#### Partitioning the Set of the Results
$$
\begin{align*}
\text{all n-bit strings} &= \text{all n-bit strings starting with 0} \\&+ \text{all n-bit strings starting with 1}
\end{align*}
$$
Lets write this in $BS(n)$
$$
\begin{align*}
BS(n) &= BS(n-1) + BS(n-1)\\
&= 2BS(n-1)\\\\
BS(0) &= 1
\end{align*}
$$
![[Pasted image 20251113164033.png]]

#### Solution
##### Unraveling
$$
\begin{align*}
BS(n) &= 2BS(n-1)\\
BS(n) &= 2[2BS(n-2)] = 2^2BS(n-2)\\
BS(n) &= 2^2[2BS(n-3)] = 2^3BS(n-3)\\
&\vdots\\
BS(n) &= 2^kBS(n-k)\\
&\vdots\\
BS(n) &= 2^nBS(n-(n))\\
&= 2^nBS(0)\\
&= \boxed{2^n}
\end{align*}
$$
##### HRRCC
Guess $BS(n) = Ar^n$
$$
\begin{align*}
BS(n) &= 2BS(n-1)\\
Ar^n &= 2Ar^{n-1}\\
\frac{r^n}{r^{n-1}} &=2\\
r &=\boxed{2}
\end{align*}
$$
Now we know 
$$
BS(n) = A2^n
$$
To find $A$, we need to use the base case
$$
\begin{align*}
BS(0) &= 1\\
A2^{(0)} &= 1\\
A &= \boxed{1}
\end{align*}
$$
Result:
$$
BS(n) = 2^n
$$

### 2 by n Domino Tiles
How many ways can we fill a $2$ by $n$ grid with dominos?
Each domino takes up $2$ adjacent squares (can be vertical or horizontal)

#### Finding the Pattern
Let $DT(n)$ be the number of different domino tilings of a $2$ by $n$ grid

| $n$ | $DT(n)$ |
| --- | ------- |
| 1   | 1       |
| 2   | 2       |
| 3   | 3       |
| 4   | 5       |
| 5   | 8       |
| 6   | 13      |
| 7   | 21      |
| 8   | 34      |
> [!WARNING]
> Sometime it is helpful to find the pattern, but need to be careful (as suggested in this problem) pattern might not be what it first appears 
> Notice for $n \leq 3$ the patterns seems to emerge as a linear pattern. However after $n=4$ the pattern emerges differently from linear relation

#### Partitioning the Set of Results
$$
\begin{align*}
\text{ways to orient dominos} &= \text{the first column has 1 vertical tile}\\
&+ \text{the first column contains 2 horizontal tiles} 
\end{align*}
$$

![[Pasted image 20251113172824.png]]

Write in $DT(n)$
$$
\begin{align*}
DT(n) &= DT(n-1) + DT(n-2)\\\\
DT(1) &= 1\\ 
DT(2) &= 2
\end{align*}
$$
> [!NOTE]
> Notice having 2 base cases
>
> The number of base cases needed **depends on how far back the recurrence goes** → what is the maximum depth the recursive call(s) go?
> With this problem, since there is a $DT(n-2)$ the number of base cases is 2

#### Solution
Guess $DT(n) = Ar^n$
$$
\begin{align*}
DT(n) &= DT(n-1) + DT(n-2)\\
Ar^n &= Ar^{n-1} + Ar^{n-2}\\
\frac{r^n}{r^{n-2}} &= r\\
r^2 &= r+1\\
r^2 - r - 1 &= 0 \to \text{characteristic polynomial, need to solve for root}\\\\
r &= \frac{-b \pm \sqrt{b^2-4ac}}{2a}\\
r &= \frac{1 \pm \sqrt{1 - 4(1)(-1)}}{2(1)}\\
r &= \boxed{\frac{1 \pm \sqrt{5}}{2}}
\end{align*}
$$

> [!PROBLEM]
> $r$ has 2 values
> To resolve this, we incorporate both roots for $DT(n)$ in the form of linear combination

Let 2 roots be:
- $\phi = \frac{1+\sqrt{5}}{2} \approx 1.618$
- $\overline{\phi} = \frac{1-\sqrt{5}}{2} = \frac{-1}{\phi} \approx -0.618$

As such
$$
DT(n) = A\phi^n + B\overline{\phi}^n
$$
It is clear that the first term $A\phi^n$ dominates the term (since $\phi > \overline{\phi}$). Therefore
$$
DT(n) \in \Theta(\phi^n) = \Theta(1.618^n)
$$

> [!NOTE]
> To find $A$ and $B$, we want to use [Gaussian Elimination](https://math.libretexts.org/Courses/Palo_Alto_College/College_Algebra/05%3A_Systems_of_Equations_and_Inequalities/5.04%3A_Solving_Systems_with_Gaussian_Elimination)
> For the purpose of *computer science* the coefficients $A$ and $B$ are not important as we only care about the growth rate of the function.
> 
> We can find $A = \frac{\phi}{\sqrt{5}}$ and $B = \frac{1}{\phi\sqrt{5}}$
> Use this, we can get the closed form of the algorithm to be
> $$DT(n) = \frac{1}{\sqrt{5}}(\phi^{n+1}+(\frac{-1}{\phi})^{n+1})$$
> 
> Notice that the second term $(\frac{-1}{\phi})^{n+1}$ grows smaller as $n$ increases, we can approximate $DT(n)$ as the *nearest integer* of $\frac{\phi^{n+1}}{\sqrt{5}}$ which we get
> $$DT(n) = \lfloor \frac{\phi^{n+1}}{\sqrt{5}} \rceil$$
> → $\lfloor x \rceil$ means the nearest integer of $x$

### N-Bit Binary Strings Without "11" Substring
#### Finding the Pattern
let $A(n)$ be the number of binary strings without "11" substring

| $n$ | $A(n)$ |
| --- | ------ |
| 0   | 1      |
| 1   | 2      |
| 2   | 3      |
| 3   | 5      |

#### Partition the set
$$
\begin{align*}
\text{All n-bit "11" avoiders} &= \text{All that starts with 0} + \text{All that starts with 1}\\
A(n) &= A(n-1) + A(n-2)\\\\
A(1) &= 2\\
A(0) &= 1
\end{align*}
$$
#### Solution
Guess $A(n) = cr^n$
$$
\begin{align*}
A(n) &= A(n-1) + A(n-2)\\
cr^n &= cr^{n-1} + cr^{n-2}\\
r^2 &= r + 1\\
r^2 -r-1 &= 0\\
r &= \frac{1 \pm \sqrt{5}}{2} \to \text{Fibonacci Sequence}
\end{align*}
$$
As such the runtime for $A(n) \in \boxed{\Theta(1.618^n)}$

### N-Bit Binary Strings Without "111" Substring
#### Finding the Pattern
Let $B(n)$ be the number of binary strings of length $n$ without a occurrence of "111"

| $n$ | $B(n)$ |
| --- | ------ |
| 0   | 1      |
| 1   | 2      |
| 2   | 4      |
| 3   | 7      |
| 4   | 13     |
#### Partition the Set
$$
\begin{align*}
\text{All n-bit "111" avoiders} &= \text{All that starts with 0} + \text{All that starts with 10} + \text{All that starts with 11}\\
B(n) &= B(n-1) + B(n-2) + B(n-3)\\\\
B(0) &= 1\\
B(1) &= 2\\
B(2) &= 4
\end{align*}
$$
#### Solution
Guess $B(n) = Ar^n$
$$
\begin{align*}
B(n) &= B(n-1) + B(n-2) + B(n-3)\\
Ar^n &= Ar^{n-1} + Ar^{n-2} + Ar^{n-3}\\
r^3 &= r^2 + r + 1\\
r^3 - r^2 - r - 1 &= 0
\end{align*}
$$
> [!NOTE]
> Notice that this is a *cubic*, to find the root we just use a online calculator to find the root
> 
> since with polynomials, it is possible to have *complex numbers* as a root and is difficult to solve a polynomial without generalized equation

By plugging in $r^3 - r^2 - r - 1$ into [Wolfram|Alpha: Computational Intelligence](https://www.wolframalpha.com/), we get 3 roots:
- $r \approx 1.8393$
- $r \approx -0.42 - 0.61i$
- $r \approx -0.42 + 0.61i$
Note that although we have 3 roots and have *complex roots*, the real root ($1.84$) is the root that dominates the runtime
$$
B(n) \in \boxed{\Theta(1.8393^n)}
$$
---
## Key Insights
- **Multiplicity**: If a root $r$ appears $m$ times, the terms are $(A_0 + A_1n + \dots + A_{m-1}n^{m-1})r^n$.
- **Dominant Root**: In CS, we usually only care about the largest root, as it dictates the **Big-$\Theta$** growth.
- **Base Cases**: You need as many base cases as the "depth" of your recurrence (e.g., $f(n-3)$ needs 3 base cases).