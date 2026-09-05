> [!ABSTRACT]
> 
> Determining the runtime of an algorithm involves analyzing the growth rate of its execution time relative to the input size $n$. We use Asymptotic Notation ($O, \Omega, \Theta$) to provide upper, lower, and tight bounds, often applying the Product and Sum rules to decompose complex loops.

---
## 1. Product Rule for Loops

If a loop runs $O(T_2(n))$ times and the body of that loop takes $O(T_1(n))$ to execute, the total time is:

$$
O(T_1(n) \cdot T_2(n))
$$

### The "Tightness" Trap
While the product rule always provides a valid upper bound ($O$), it is not always a **tight bound** ($\Theta$). A pessimistic upper bound occurs when the "worst case" of the inner logic rarely happens across all iterations of the outer loop.
#### Example: Disjoint Set (Two-Pointer Method)
To check if two sorted lists are disjoint (no common elements), we use two pointers, $i$ and $j$.
- **Pessimistic View**: The outer loop runs $n$ times, and the inner logic seems to "interact" with $n$ elements. One might guess $O(n^2)$.

![[Pasted image 20251108140656.png]]

- **Actual Runtime**: In each comparison, either $i$ increments or $j$ increments. Neither pointer ever resets. Since each pointer can only move $n$ times, the total number of operations is at most $2n$.

![[Pasted image 20251108141805.png]]

- **Result**: The runtime is $\Theta(n)$, even though a naive product rule might suggest higher.

---
## 2. Sum Rule for Processes
If an algorithm performs two distinct tasks sequentially—first Process A then Process B—the total runtime is the sum of their individual runtimes:

$$
O(T_1(n) + T_2(n))
$$

In asymptotic analysis, this simplifies to the maximum of the two: $O(\max(T_1(n), T_2(n)))$.

---
## 3. Proving Tight Bounds ($\Theta$)
To conclude that an algorithm is $\Theta(f(n))$, you must prove both the upper and lower bounds.
### Example: Selection Sort Comparisons
Consider the nested loop structure of Selection Sort (MinSort):

![[Pasted image 20251107171424.png]]

1. Lower Bound ($\Omega$): By counting the number of comparisons:
    $$
    (n-1) + (n-2) + \dots + 1 = \frac{n(n-1)}{2}
    $$
    
    This is a polynomial of degree 2, so $T(n) \in \Omega(n^2)$.
2. **Upper Bound ($O$):** By the **Product Rule**, the outer loop runs $n$ times and the inner loop runs at most $n$ times. Thus, $T(n) \in O(n^2)$.

> [!IMPORTANT]
> 
> Because $T(n) \in \Omega(n^2)$ AND $T(n) \in O(n^2)$, we can state definitively that:
> 
> $$
> T(n) = \Theta(n^2)
> $$

---
## 4. Analysis Summary

|**Rule**|**Mathematical Form**|**Common Usage**|
|---|---|---|
|**Product Rule**|$O(T_1 \cdot T_2)$|Nested loops, multiplying iterations by body cost.|
|**Sum Rule**|$O(T_1 + T_2)$|Sequential blocks of code or function calls.|
|**Tight Bound**|$\Theta(f(n))$|When $O$ and $\Omega$ meet, describing the exact growth rate.|
