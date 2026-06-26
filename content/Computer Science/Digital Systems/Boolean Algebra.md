## 1. Logic Operators

In Boolean algebra, every variable and operator returns a binary value of either $0$ or $1$.
### Basic Operators

These are the foundational building blocks of all logical circuits.

| **Set Operator Title** | **Algebraic Expression** | **Logical Representation** | **Logic Gate Symbol**                |
| ---------------------- | ------------------------ | -------------------------- | ------------------------------------ |
| **Intersection**       | $A \cdot B$              | **AND**                    | ![[Pasted image 20260111120532.png]] |
| **Union**              | $A + B$                  | **OR**                     | ![[Pasted image 20260111115910.png]] |
| **Complementary**      | $A'$ or $\overline{A}$   | **NOT**                    | ![[Pasted image 20260111115925.png]] |
| **Identity**           | $A$                      | **BUFFER**                 |                                      |
### Derived Operators

These operators are combinations of the basic operators, often used to simplify circuit design.

| **Set Operator Title**   | **Algebraic Expression** | **Logical Representation** | **Logic Gate Symbol**                |
| ------------------------ | ------------------------ | -------------------------- | ------------------------------------ |
| **Alternative Denial**   | $(A \cdot B)'$           | **NAND**                   | ![[Pasted image 20260111115951.png]] |
| **Joint Denial**         | $(A + B)'$               | **NOR**                    | ![[Pasted image 20260111120002.png]] |
| **Symmetric Difference** | $A \oplus B$             | **XOR**                    |                                      |
| **Equivalence**          | $(A \oplus B)'$          | **XNOR**                   |                                      |

---

## 2. Axioms and Theorems

Boolean logic is defined by a set of axioms (assumed truths) and theorems (proven rules). Every rule has a **Dual**, which is equally valid.
### Axioms

The fundamental assumptions of the binary field.

|**Name**|**Axiom**|**Dual**|
|---|---|---|
|**Binary Field**|$B = 0$ if $B \neq 1$|$B = 1$ if $B \neq 0$|
|**NOT**|$\overline{0} = 1$|$\overline{1} = 0$|
|**AND / OR**|$0 \cdot 0 = 0$|$1 + 1 = 1$|
|**AND / OR**|$1 \cdot 1 = 1$|$0 + 0 = 0$|
|**AND / OR**|$0 \cdot 1 = 1 \cdot 0 = 0$|$1 + 0 = 0 + 1 = 1$|
### Theorems

Rules used for simplifying Boolean expressions.

|**Name**|**Theorem**|**Dual**|
|---|---|---|
|**Identity**|$B \cdot 1 = B$|$B + 0 = B$|
|**Null Element**|$B \cdot 0 = 0$|$B + 1 = 1$|
|**Idempotency**|$B \cdot B = B$|$B + B = B$|
|**Involution**|$\overline{\overline{B}} = B$|$\overline{\overline{B}} = B$|
|**Complements**|$B \cdot \overline{B} = 0$|$B + \overline{B} = 1$|
|**Commutativity**|$B \cdot C = C \cdot B$|$B + C = C + B$|
|**Associativity**|$(B \cdot C) \cdot D = B \cdot (C \cdot D)$|$(B + C) + D = B + (C + D)$|
|**Distributing**|$(B \cdot C) + (B \cdot D) = B \cdot (C + D)$|$(B + C) \cdot (B + D) = B + (C \cdot D)$|
|**Covering**|$B \cdot (B + C) = B$|$B + (B \cdot C) = B$|
|**Combining**|$(B \cdot C) + (B \cdot \overline{C}) = B$|$(B + C) \cdot (B + \overline{C}) = B$|
|**Consensus**|$(B \cdot C) + (\overline{B} \cdot D) + (C \cdot D) = (B \cdot C) + (\overline{B} \cdot D)$|$(B + C) \cdot (\overline{B} + D) \cdot (C + D) = (B + C) \cdot (\overline{B} + D)$|
|**De Morgan's**|$\overline{B_0 \cdot B_1 \cdot \dots} = (\overline{B_0} + \overline{B_1} + \dots)$|$\overline{B_0 + B_1 + \dots} = (\overline{B_0} \cdot \overline{B_1} \cdot \dots)$|

---
## 3. Boolean Duality

Duality is a central property of Boolean algebra. A dual expression is derived by replacing:
- $\cdot$ (AND) with $+$ (OR)
- $+$ (OR) with $\cdot$ (AND)
- $0$ with $1$
- $1$ with $0$

**Generalized Duality Principle:**

$$f(X_1, X_2, \dots, X_n, 0, 1, +, \cdot) \iff f(X_1, X_2, \dots, X_n, 1, 0, \cdot, +)$$

> [!NOTE]
> 
> The Duality Principle states that any theorem that can be proven is automatically proven for its dual.
> 
> > [!DANGER] Warning
> > 
> > Duality is not the same as De Morgan's Law. Duality swaps operators and constants but does not complement the individual variables.