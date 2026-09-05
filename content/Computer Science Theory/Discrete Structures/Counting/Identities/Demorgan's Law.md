> [!ABSTRACT]
> 
> De Morgan's Laws describe the relationship between the complement of the union (or intersection) of sets and the intersection (or union) of their individual complements. In discrete mathematics, these laws serve as the logical bridge that allows us to use the Principle of Inclusion-Exclusion to solve "None of" or "Neither" counting problems.

---
## 1. The Two Laws
De Morgan's Laws state that the complement of a group operation is equivalent to the operation on the individual complements, provided the operator is flipped ($\cup$ becomes $ \cap $, and $\cap$ becomes $\cup$).
### The Complement of the Union

$$
\overline{A \cup B} = \overline{A} \cap \overline{B}
$$

- **Meaning**: The set of elements that are **not** in $A$ or $B$ is exactly the set of elements that are **not** in $A$ **AND** **not** in $B$.
### The Complement of the Intersection

$$
\overline{A \cap B} = \overline{A} \cup \overline{B}
$$

- **Meaning**: The set of elements that are **not** in both $A$ and $B$ is the set of elements that are **not** in $A$ **OR** **not** in $B$.

---
## 2. The Bridge to Inclusion-Exclusion
In combinatorics, we are often asked to count elements that satisfy **none** of a set of conditions (properties). De Morgan's Law is what transforms these "None" problems into "Union" problems that we can actually solve.
### The Problem Transformation
If we have a universal set $S$ and properties $P_1, P_2, \dots, P_n$, let $A_i$ be the set of elements having property $P_i$.

We want to find the number of elements with none of the properties:

$$
N(P_1' P_2' \dots P_n') = |\overline{A_1} \cap \overline{A_2} \cap \dots \cap \overline{A_n}|
$$

By De Morgan's Law, this intersection of complements is equal to the complement of the union:

$$
|\overline{A_1} \cap \overline{A_2} \cap \dots \cap \overline{A_n}| = |\overline{A_1 \cup A_2 \cup \dots \cup A_n}|
$$

### The Final Formula
Since the size of a complement is simply the total minus the set itself:

$$
|\overline{A_1 \cup A_2 \cup \dots \cup A_n}| = |S| - |A_1 \cup A_2 \cup \dots \cup A_n|
$$

> [!IMPORTANT]
> This allows us to use the **[[Inclusion Exclusion|Inclusion-Exclusion Principle]]** to calculate $|A_1 \cup A_2 \cup \dots \cup A_n|$ and subtract it from the total.

---
## 3. Example: Counting Integers
**Goal**: Find how many integers between 1 and 100 are divisible by **neither** 2 **nor** 5.
1. **Define Sets**:
    - $S = \{1, \dots, 100\}, |S| = 100$
    - $A = \{n \in S : 2|n\}$
    - $B = \{n \in S : 5|n\}$
2. **Target**: We want "Neither 2 nor 5", which is $|\overline{A} \cap \overline{B}|$.
3. **Apply De Morgan**: $|\overline{A} \cap \overline{B}| = |\overline{A \cup B}| = |S| - |A \cup B|$.
4. **Inclusion-Exclusion**:
    - $|A| = 50$
    - $|B| = 20$
    - $|A \cap B| = 10$ (divisible by 10)
    - $|A \cup B| = 50 + 20 - 10 = 60$.
5. **Result**: $100 - 60 = 40$.

---
## 4. Logical Negation
De Morgan's Laws are also used in logic to negate statements involving "And" ($\land$) and "Or" ($\lor$):
- $\neg(P \lor Q) \equiv \neg P \land \neg Q$
- $\neg(P \land Q) \equiv \neg P \lor \neg Q$