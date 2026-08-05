---
tags:
  - Counting
  - Inclusion-Exclusion
  - Complement
  - DeMorgan
  - Power-Rule
---
$$
\begin{align*}
|S_1 \cup S_2 \cup \dots \cup S_n| = &\sum_{1 \leq i \leq n} |S_i| \\
- &\sum_{1 \leq i < j \leq n} |S_i \cap S_j| \\
+ &\sum_{1 \leq i < j < k \leq n} |S_i \cap S_j \cap S_k| \\
- &\dots + (-1)^{n+1} |S_1 \cap S_2 \cap \dots \cap S_n|
\end{align*}
$$

![[Pasted image 20251005191903.png]]

---

### Examples

#### 1. Total Number of Strings

How many $n$-length strings are there over an alphabet of size $X$?

$$
X^n \quad \text{(Power Rule)}
$$

---

#### 2. Strings Where Each Character Is Used at Least Once

Let the alphabet be $\{a, b, c\}$ and string length be 12.  
We want to count strings where **each character appears at least once**.

This is equivalent to:

$$
\sum_{k=0}^{x} \binom{x}{k} (x - k)^n (-1)^k
$$

For $x = 3$ and $n = 12$:

$$
\sum_{k=0}^{3} \binom{3}{k} (3 - k)^{12} (-1)^k
$$

---

### Set Interpretation

Let:
- $A$: strings where `a` is used at least once  
- $B$: strings where `b` is used at least once  
- $C$: strings where `c` is used at least once  

We want:

$$
|A \cap B \cap C|
$$

Using [[Demorgan's Law]]:

$$
\begin{align*}
|\overline{A \cap B \cap C}| &= |\overline{A} \cup \overline{B} \cup \overline{C}| \\
|A \cap B \cap C| &= |\text{ALL}| - |\overline{A} \cup \overline{B} \cup \overline{C}|
\end{align*}
$$

Expanding:

$$
\begin{align*}
|\text{ALL}| - [&|\overline{A}| + |\overline{B}| + |\overline{C}| \\
&- |\overline{A} \cap \overline{B}| - |\overline{A} \cap \overline{C}| - |\overline{B} \cap \overline{C}| \\
&+ |\overline{A} \cap \overline{B} \cap \overline{C}|]
\end{align*}
$$

Where:
- $\overline{A}$ = strings without `a`
- $\overline{B}$ = strings without `b`
- $\overline{C}$ = strings without `c`

Final result:

$$
\sum_{k=0}^{3} \binom{3}{k} (3 - k)^{12} (-1)^k
$$

See [[Set Theory]] for more information