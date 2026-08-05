> [!ABSTRACT]
> 
> A set is an unordered collection of distinct objects, called elements. Set theory provides the mathematical foundation for almost all other areas of mathematics and is the basis for data structures in computer science.

---
## 1. Core Operations
Operations on sets allow us to combine or compare collections of data.
### Union ($A \cup B$)
- **Keyword**: **OR**
- **Definition**: The set containing all elements that are in $A$, or in $B$, or in both.
- **Logic**: $x \in A \cup B \iff (x \in A \lor x \in B)$
### Intersection ($A \cap B$)
- **Keyword**: **AND**
- **Definition**: The set containing only elements that are in both $A$ and $B$.
- **Logic**: $x \in A \cap B \iff (x \in A \land x \in B)$
### Difference ($A \setminus B$ or $A - B$)
- **Keyword**: **NOT IN**
- **Definition**: The set of elements that are in $A$ but are **not** in $B$.
- **Logic**: $x \in A \setminus B \iff (x \in A \land x \notin B)$

![Image of Venn diagrams for Union, Intersection, and Difference](https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcTKlCiM0VyE9fJ7M5txToT9F9Ah0EPDj-lQcWMoI6HBxY6EEdFSDf0cKd3ClV5BY1iQXSNE4GiEKmwL5qEexPx4haTqSyjnu-wQfFJupPINcato26o)

---
## 2. Relationships and Subsets
### Subset ($A \subseteq B$)
- **Definition**: Every element of $A$ is also an element of $B$.
- **Note**: If $A$ is a subset of $B$ but $A \neq B$, it is a **proper subset** ($A \subset B$).
### Complement ($\overline{A}$ or $A^c$)
- **Definition**: The set of all elements in the Universal Set ($U$) that are not in $A$.
- **Identity**: $\overline{A} = U \setminus A$.
- **Key Concept**: See [[Demorgan's Law]] for how complements interact with Union and Intersection.

---
## 3. Counting and Combinations

> [!INFO]
> 
> The number of elements in a set is called its cardinality, denoted $|A|$.

### Subsets and Combinations
To find the number of ways to choose a subset of size $k$ from a set of size $n$, we use **"n choose k"**:
- **Notation**: $C(n, k)$ or $\binom{n}{k}$
- **Formula**: $\binom{n}{k} = \frac{n!}{k!(n-k)!}$
### The Power Set ($\mathcal{P}(A)$)
- **Definition**: The set of all possible subsets of $A$.
- **Cardinality**: If $|A| = n$, then $|\mathcal{P}(A)| = 2^n$.
- **Identity**: This is proven via the [[Sum Identity]].

---
## 4. Fundamental Identities

| **Name**                | **Identity**                                           |
| ----------------------- | ------------------------------------------------------ |
| **Inclusion-Exclusion** | $$\|A \cup B\| = \|A\| + \|B\| - \|A \cap B\|$$        |
| **De Morgan's (I)**     | $\overline{A \cup B} = \overline{A} \cap \overline{B}$ |
| **De Morgan's (II)**    | $\overline{A \cap B} = \overline{A} \cup \overline{B}$ |
