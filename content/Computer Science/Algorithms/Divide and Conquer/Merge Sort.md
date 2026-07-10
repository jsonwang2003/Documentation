```pseudo
	\begin{algorithm}
	\caption{Merge Sort}
	\begin{algorithmic}
	\Input array to be sorted
	\Output sorted array
	\Procedure{mergesort}{$a[1 \dots n]$}
		\If{$n > 1$}
			\State $ML = mergesort(a[1 \dots \lfloor \frac{n}{2} \rfloor])$
			\State $MR = mergesort(a[\lfloor \frac{n}{2} + 1, \dots n])$
			\Return $merge(ML, MR)$
		\Else
			\Return $a$
        \EndIf
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

Suppose mergesort runs in $T(n)$ time for inputs of length $n$, then each recursive call runs in $T\left( \frac{n}{2} \right)$ time and merge runs in $O(k + \ell)$ time where $k, \ell = \frac{n}{2}$. so merge runs in $O(n)$ time

$$
	\begin{align*}
	T(n) &= 2T\left( \frac{n}{2} \right) + O(n)\\
	&= \boxed{O(n\log n)}
	\end{align*}
$$

---
# Correctness
Base case: $n=1$ mergesort returns the original array $a$ which is trivially sorted
**Inductive Hypothesis**: Suppose that for some $n > 1$, $mergesort(a[1\dots k])$ outputs the elements of $a$ in sorted order on all inputs of size $k$ where $1 \leq k < n$ we want to show that it works for inputs of size $n$

Since $n > 1$, $mergesort(a[1\dots n])$ returns $merge(ML, MR)$ where $ML = mergesort\left( a\left[ 1, \dots \left\lfloor  \frac{n}{2}  \right\rfloor \right] \right)$ and $MR = mergesort\left( a\left[ \left\lfloor  \frac{n}{2}  \right\rfloor + 1, \dots n+1 \right] \right)$

Since $\left\lfloor  \frac{n}{2}  \right\rfloor < n$, the inductive hypothesis ensures that $ML$ and $MR$ are sorted. And merge combines two sorted lists so the algorithm returns the elements in sorted order
