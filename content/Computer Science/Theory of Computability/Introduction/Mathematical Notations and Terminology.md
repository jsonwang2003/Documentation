## 1. Sets
A **set** is a group of objects represented as a unit.
- **Elements:** The objects within a set. The symbols $\in$ and $\not\in$ denote membership and non-membership.
- **Subsets:** $A \subseteq B$ means every member of $A$ is also in $B$. 
- **Proper Subset**: $A \subsetneq B$  means $A \subseteq B$ but $A \neq B$.
- **Infinite Sets:** Sets with infinitely many elements, often denoted with "$\dots$" 
	- Natural numbers → $\mathbb{N} = \{ 1 ,2,3,\dots \}$
	- Integers → $\mathbb{Z}=\{ \dots, -2, -1, 0, 1, 2, \dots \}$   
- **Special Sets:**
    - **Empty Set ($\emptyset$):** A set with zero members.
    - **Multiset:** A set where the number of occurrences of an element matters.
	    - $\{ 7 \} \neq \{ 7, 7 \}$
- **Set Operations:**
    - **Union ($A \cup B$):** Combines all elements from both sets.
    - **Intersection ($A \cap B$):** Elements common to both sets.
    - **Complement ($\bar{A}$):** All elements under consideration that are _not_ in A.

| Union                                | Intersection                              |
| ------------------------------------ | ----------------------------------------- |
| ![[Pasted image 20260404212820.png]] | ![[Pasted image 20260404212832.png\|169]] |
## 2. Sequences and Tuples

A **sequence** is a list of objects in a specific order, designated by parentheses.
- **Order and Repetition:** Unlike sets, both order and repetition matter in sequences 
	- $(7, 21, 57) \neq (57, 7, 21)$
- **Tuples:** Finite sequences. A $k$-tuple has $k$ elements; a 2-tuple is an **ordered pair**.
- **Power Set:** The set of all subsets of a given set.
## 3. Functions and Relations
- **Functions:** A rule that maps an input (from a **domain** $D$) to exactly one output (from a **range** $R$ or codomain). 
- **Relations:** A set of ordered pairs representing a property between elements (e.g., "less than" on integers).
- **Equivalence Relation:** A special type of relation that is reflexive, symmetric, and transitive.
## 4. Graphs

A graph consists of a set of **nodes** (vertices) and **edges** connecting them.
- **Directed vs. Undirected:** Directed edges have an orientation $(u \to v)$; undirected edges do not.
- **Paths and Cycles:** A **path** is a sequence of nodes connected by edges; a **cycle** is a path that starts and ends at the same node.
- **Connectedness:** An undirected graph is **connected** if every pair of nodes has a path between them.
## 5. Strings and Languages
- **Alphabet ($\Sigma$):** Any finite, non-empty set of symbols.
- **String:** A finite sequence of symbols from an alphabet.
    - **Length:** The number of symbols in a string.
    - **Empty String ($\epsilon$):** A string of length 0.
- **Language:** A set of strings.
## 6. Boolean Logic
A system for manipulating the values **TRUE** and **FALSE** (often represented as 1 and 0).
- **Operations:**
    - **Negation ($\neg$):** NOT.
    - **Conjunction ($\land$):** AND.
    - **Disjunction ($\lor$):** OR.
    - **Implication ($\to$):** IF-THEN.