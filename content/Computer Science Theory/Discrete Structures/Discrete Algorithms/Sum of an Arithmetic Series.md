An **Arithmetic Series** is a sum over a sequence of numbers starting at $a_0$ that jump by a common difference $d$ and that have $N$ terms

$$
\begin{align*}
\sum_{i=0}^{n} a_0 + d_i &= N(\frac{\text{first} + \text{last}}{2})\\
&= (n + 1)(\frac{a_0 + (a_0 + d_n)}{2})
\end{align*}
$$

## Process
1. Add in another copy of the sum and reverse the direction.
2. Group each term together to sum to *first* + *last*
3. There are $n$ pairs

$$
\begin{align*}
1 + 2 + ... + (n-1) + (n) &= \sum_{k=1}^{n}k\\
n + (n-1) + ... + 2 + 1 &= \sum_{k=1}^n k\\
(n+1) + (n+1) + ... + (n+1) + (n+1) &= 2\sum_{k=1}^{n}k\\
(n+1)n &= 2\sum_{k=1}^{n}k\\
\frac{n(n+1)}{2} &= \sum_{k=1}^{n}k
\end{align*}
$$

## Example

| Examples                      | *First* | *Last* | $d$ | $N$ | *Sum*                   |
| ----------------------------- | ------- | ------ | --- | --- | ----------------------- |
| $1+2+3+4+5+6+7+8+9+10$        | 1       | 10     | 1   | 10  | $\frac{11 \cdot 10}{2}$ |
| $1+3+5+7+9+11+13+15$          | 1       | 15     | 2   | 8   | $\frac{16 \cdot 8}{2}$  |
| $21+31+41+51+61+71+81+91+101$ | 21      | 101    | 10  | 9   | $\frac{122 \cdot 9}{2}$ |
