> [!ABSTRACT]
> 
> The fundamental difference between algorithm types lies in their consistency and reliability. While Deterministic algorithms offer a predictable path, Randomized algorithms leverage probability to optimize for speed or simplicity, trading off either consistent runtime (Las Vegas) or guaranteed correctness (Monte Carlo).

---
## 1. Deterministic Algorithms
A deterministic algorithm is a "pure function" of its input.
- **Consistency**: Given input $X$, it will always follow the same execution path and produce the same output $Y$.
- **Predictability**: The runtime is fixed for any specific input.
- **Examples**: Selection Sort, Merge Sort, and Binary Search.

---
## 2. Randomized Algorithms
These algorithms accept a second hidden input: a **source of random bits**. This randomness allows the algorithm to make "choices" during execution.
### A. Las Vegas Algorithms
"Always correct, sometimes slow."

A Las Vegas algorithm uses randomness to avoid "bad" cases. It will never give you the wrong answer, but you cannot guarantee exactly how long it will take.

#### Quicksort (Randomized)

Quicksort is the quintessential Las Vegas algorithm. By picking a **random pivot**, we ensure that no specific input (like an already sorted list) can consistently trigger the $O(n^2)$ worst-case.

```pseudo
\begin{algorithm}
\caption{Quicksort}
\begin{algorithmic}
\Procedure{Quicksort}{$[a_1, a_2, \dots, a_n]$: a list of size $n$}
	\If{$n=1$}
		\Return $a_1$
    \EndIf
    \State Pick a random index $1 \leq i \leq n$
    \State Partition the list into $SL, Sv, SR$
    \State $L = Quicksort(SL)$
	\State $R = Quicksort(SR)$
	\Return $L \circ Sv \circ R$
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

- **Logic**:
    1. Pick a random element as a **Pivot**.
    2. **Partition** the array: elements $<$ pivot to the left, elements $>$ pivot to the right.
    3. **Recurse** on both halves.
- **Runtime Complexity**:
    - **Best/Average Case**: $\Theta(n \log n)$ — Occurs when the pivot splits the list roughly in half.
    - **Worst Case**: $O(n^2)$ — Occurs if the pivot is always the extreme (min/max), but the probability of this happening with random pivots is nearly zero for large $n$.
### B. Monte Carlo Algorithms
"Always fast, sometimes wrong."

A Monte Carlo algorithm has a fixed execution time, but there is a small, calculable probability that the output is incorrect. We usually "boost" the correctness by running the algorithm multiple times.

```pseudo
	\begin{algorithm}
	\caption{Monte Carlo Algorithm}
	\begin{algorithmic}
	\Procedure{MonteCarloAlgorithm}{$x$: input, $n$: iterations}
		\State Success = 0
		\For{$i=1$ to $n$}
			\State sample = GenerateRandomSample($x$)
			\If{PropertyCheck(sample) == \True}
				\State Success = Success + 1
            \EndIf
        \EndFor
        \Return Success / $n$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

- **Logic**: It samples the search space. The more time you give it (more iterations), the higher the probability of success.
- **Example**: **Primality Testing (Miller-Rabin)**. It can tell you if a number is prime very quickly, but there is a tiny chance it labels a composite number as prime.
- **Example**: **Estimating $\pi$** by randomly dropping needles on a grid.

---
## 3. Comparison Summary

|**Feature**|**Deterministic**|**Las Vegas (Randomized)**|**Monte Carlo (Randomized)**|
|---|---|---|---|
|**Output**|Always Correct|**Always Correct**|**Probability of Error**|
|**Runtime**|Fixed for input $X$|**Variable (Random)**|**Fixed/Deterministic**|
|**Primary Goal**|Reliability|Avoid Worst-Case Scenarios|Speed in massive search spaces|
|**Example**|Merge Sort|Randomized Quicksort|Miller-Rabin Primality Test|

---
## 4. Why Use Randomness?
1. **Symmetry Breaking**: In distributed systems, randomness helps multiple computers decide who goes first without a central leader.
2. **Defeating Adversaries**: If an attacker knows your deterministic pivot-picking strategy, they can give you an input that makes your server crash ($O(n^2)$). Randomization makes your "weakness" impossible to predict.
3. **Simplicity**: Often, a randomized algorithm is much shorter and easier to implement than a complex deterministic one that achieves the same average runtime.