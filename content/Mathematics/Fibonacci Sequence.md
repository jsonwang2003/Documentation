The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with **0** and **1**.
### Mathematical Definition

The sequence is defined by the recurrence relation:

$$F_n = F_{n-1} + F_{n-2}$$

With the base cases:
- $F_0 = 0$
- $F_1 = 1$

**First 10 terms:** 0, 1, 1, 2, 3, 5, 8, 13, 21, 34...

---
### Implementation Methods

#### 1. Recursive (Naive)

This approach directly follows the mathematical definition.

```cpp
int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}
```

- **Time Complexity:** $O(2^n)$ — Exponential growth due to redundant calculations.
- **Space Complexity:** $O(n)$ — Maximum depth of the recursion stack.
#### 2. Dynamic Programming (Iterative)

By storing or calculating previous values in a loop, we avoid redundant work.

```cpp
int fib(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1, sum;
    for (int i = 2; i <= n; i++) {
        sum = a + b;
        a = b;
        b = sum;
    }
    return b;
}
```

- **Time Complexity:** $O(n)$ — Linear time.
- **Space Complexity:** $O(1)$ — Only a few variables are used.

---
### Properties and The Golden Ratio

As $n$ approaches infinity, the ratio of successive Fibonacci numbers ($\frac{F_{n}}{F_{n-1}}$) converges to the **Golden Ratio** ($\phi$):

$$\phi = \frac{1 + \sqrt{5}}{2} \approx 1.618$$

---
### Complexity Summary

|**Method**|**Time Complexity**|**Space Complexity**|
|---|---|---|
|**Naive Recursion**|$O(2^n)$|$O(n)$|
|**Memoization**|$O(n)$|$O(n)$|
|**Iterative (DP)**|$O(n)$|$O(1)$|
|**Matrix Exponentiation**|$O(\log n)$|$O(\log n)$|