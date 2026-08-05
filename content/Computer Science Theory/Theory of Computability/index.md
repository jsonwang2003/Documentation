---
title: Theory of Computability
---
## Main Focus
### What is a Computational Problem?
**Problem**: With some input given paired with a goal that can be determined by such input
**Solutions**: Algorithms
### How to Specify Problems?
#### Strings
**Alphabet**: A Finite Set of Symbols ($\Sigma = \{a, b, c\}$)
**String**: A Finite Sequence of Symbols ("$abba$")
- $|s| =$ length of a string $s$
**Language**: A Set of *Strings* (over a fixed $\Sigma$)
- $|L|=$ size of Language $L$ 
- $\Sigma^*=$ Set of all strings over $\Sigma$
**Regular Expression**: A way to describe the properties we want to test on
- $R = a (a \in \Sigma) \ \ \ \ \ L(a) = \{a\}$
- $R = \epsilon =$ ""          $L(\epsilon) = \{\epsilon\}$
- $R = \emptyset$                  $L(\emptyset) = \emptyset = \{\}$
##### Rules of Regular Expressions
$$
\begin{align*}
R &= R_1 \cup R_2 &L(R_1 \cup R_2) = L(R_1) \cup L(R_2)\\
R &= R_1 \cdot R_2 &L(R_1 \cdot R_2) = \{w_1w_2 | w_1 \in L(R_1), w_2 \in L(R_2)\}\\
R &= R_1^* &L(R_1^*) = \{w_1, w_2, ..., w_k | k \geq 0, \forall i, w_i \in L(R_1)\}
\end{align*}
$$

#### Numbers
- $A = n$
- $A = A_1 + A_2$
- $A = A_1 \cdot A_2$
- $A = A_1^2$
- $A = (A_1)$

Eval: $A$ → $\mathbb{N}$


> [!Example]
> $R = (a \cdot (a \cup b)^* \cdot b) \cup (b \cdot (a \cup b)^* \cdot b)$
> → "All strings over $\Sigma = {a, b}$ that begin and end with different symbols"

$$
w = \underbrace{\underbrace{aba}_{string} \in \underbrace{L_1}_{language}}_{boolean}
$$

Question:
	Given: $R$, $w$
	Output: $w \in L(R)$?

## Deterministic Finite Automaton (DFA)
> [!Abstract] Definition
> A DFA is a **5-tuple**
> $M = (Q, \Sigma, \delta, s, F)$ where:
> - $Q$ is a **finite set** of states
> - $\Sigma$ is an alphabet (finite set of symbols)
> - $\delta$: $Q \times \Sigma \to Q$ as a table of which state $q$ have an edge to with a label of $a$
> 	- $\delta(q, a) \in Q$: the function returns the destination state in $Q$
> - $s \in Q$ the starter state in $Q$
> - $F \subset Q$ the finish state(s)

The computation of DFA $M$ on input $w \in \Sigma^*$ is the sequence $q_{o}, w_{1}, q_{1}, w_{2}, q_{2}, \dots, w_{n}, q_{n}$ of $q_{i} \in Q$ and $w_{i} \in \Sigma$ such that
- $w = w_{1}w_{2}w_{3}\dots w_{n}$
- $q_{0} = s$
- $q_{i+1} = \delta(q_i, w_{i+1})$

There exist only 1 possible sequence

The computation is **accepting** if $q_{n} \in F$
The computation is **rejecting** if $q_{n} \not\in F$


$\delta^*$: $Q\times\Sigma^* \to Q$
	$\delta^*(q, \epsilon) = q$
	$\delta^*(q, aw) = \delta^*(\delta(q, a), w)$
$L(M) = {w \in \Sigma^* | \delta^*(s, w) \in F)}$


### Are Some Problems Easier / Harder to Solve?
### Are There Unsolvable Problems?





