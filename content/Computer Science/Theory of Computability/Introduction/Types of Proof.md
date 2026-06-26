## Proof by Construction
- Many theorems state that a **particular type of object exists**
- Prove by **demonstrating how to construct the object**

> [!Example] For each even number $n$ greater than $2$, there exists a $3$-regular graph with $n$ nodes.
> ### Proof:
> Let $n$ be an even number greater than $2$. Construct $G = (V, E)$ with $n$ nodes as follows. The set of nodes of $G$ is $V = \{ 0, 1, \dots, n-1 \}$, and the set of edges of $G$ is the set
> $$
> \begin{align*}
> E &= \{ \{ i, i+1 \} | \text{ for } 0 \leq i \leq n-2 \} \cup \{ \{ n-1, 0 \} \}\\
> &= \cup \left\{  \left\{  i, \frac{i+n}{2}  \right\} | \text{ for } 0 \leq i \leq \frac{n}{2} - 1  \right\}
> \end{align*}
> $$
> 
> Picture the nodes of this graph written consecutively around the circumference of a circle. 
> - The edges described in the top line of $E$ go **between adjacent pairs around the circle**. 
> - The edges described in the bottom line of $E$ go between **nodes on opposite sides of the circle**
> 
> As such every node in $G$ has degree of $3$

## Proof by Contradiction
Assume that the theorem is **false** and then show that this assumption leads to an *obviously false consequence* (contradiction)

> [!Example] $\sqrt{ 2 }$ is irrational
> ### Proof:
> Assume that $\sqrt{ 2 }$ is rational
> $$
> \sqrt{ 2 } = \frac{m}{n}
> $$
> where $m, n \in \mathbb{Z}$
> 
> If both $m$ and $n$ are divisible by the same $\mathbb{Z}$ greater than $1$, divide both by the largest such integer. Doing so doesn't change the value of the fraction.
> 
> Now, at least one of $m$ and $n$ must be an odd number
> 1. Multiply both sides of the equation by $n$ and obtain:
> $$
> n\sqrt{ 2 } = m
> $$
> 2. Square both sides:
>    $$
>	2n^2 = m^2
>	$$
>
> Since $m^2$ is $2$ times the integer $n^2$, we know that $m^2$ is even. Therefore $m$ too is even as the square of an odd number is always odd. So $m = 2k$ for some integer $k$
> $$
> \begin{align*}
> 2n^2 &= (2k)^2\\
> &= 4k^2\\
> n^2 &= 2k^2
> \end{align*}
> $$
> This shows, however, that $n^2$ is even and hence that $n$ is even. Thus we have established that both $n$ and $m$ are even. But earlier was reduced that $m$ and $n$ are not **both** even → contradiction

## Proof by Induction
Advanced method used to show that **all elements of an infinite set have a specified property**

Consist of 2 parts:
1. **Basis**: Proves that $P(1)$ is true
2. **Induction Step**: Proves that for each $i \geq 1$, if $P(i)$ is true, then so is $P(i+1)$

### Format of Induction Proof
$$
\begin{align*}
\text{Prove that } P(1) &\text{ is true}\\
&\vdots\\
\text{For each } i \geq 1 \text{, assume that } P(i) &\text{ is true and use this}\\ \text{ assumption to show that } P(i+1) &\text{ is true}\\
&\vdots
\end{align*}
$$
 
> [!Example] For each $t \geq 0$, $P_{t} = PM^t - Y\left( \frac{M^t - 1}{M-1} \right)$
> ### Proof:
> **Basis**: Prove that the formula is true for $t=0$. If $t = 0$, then the formula states that:
> $$
> P_{0} = PM^0 - Y\left( \frac{M^0-1}{M-1} \right)
> $$
> Simplify *right-hand side* by observing $M^0 = 1$
> $$
> P_{0} = P
> $$
> which holds because we have defined $P_{0}$ to be $P$. Therefore the basis of the induction is true
> 
> **Induction Step**: For each $k\geq{0}$, assume that the formula is trie for $t=k$ and show that it is true for $t = k+1$. The induction hypothesis states that:
> $$
> P_{k} = PM^k - Y\left( \frac{M^k-1}{M-1} \right)
> $$
> Our objective is to prove that
> $$
> P_{k+1} = PM^{k+1} - \left( \frac{Y((M^{k+1}-1))}{M-1} \right)
> $$
> We do so with the following steps
> 1. From the definition of $P_{k+1}$ from $P_{k}$, we know that
> $$
> P_{k+1} = P_{k}M-Y
> $$
> 2. Therefore, using the induction hypothesis to calculate $P_{k}$
> $$
> P_{k+1} = \left[ PM^k - Y\left( \frac{M^k-1}{M-1} \right) \right]M-Y
> $$
> 3. Multiplying through by $M$ and rewriting $Y$ yields
> $$
> \begin{align*}
> P_{k+1} &= PM^{k+1} - Y\left( \frac{M^{k+1}-M}{M-1} \right)-Y\left( \frac{M-1}{M-1} \right)\\
> &= PM^{k+1} - Y\left( \frac{M^{k+1}-1}{M-1} \right)
> \end{align*}
> $$
> 4. Thus the formula is correct for $t = k+1$, which proves the theorem

