## Power Rule

> [!INFO]
> For any set \( A \),  
> $$
> |\underbrace{A \times A \times A \dots \times A}_{n}| = |A|^n
> $$

### Uses
- To count the **number of strings of length \( n \)** over a **finite alphabet \( A \)**
- To count the **number of \( n \)-length sequences** using only the numbers:
	- $\{0, 1, \dots, |A| - 1\}$
	- $\{1, 2, \dots, |A|\}$
	- Any other set with **cardinality equal to $|A|$**
- To count the **number of ways of distributing $|A|$** distinct objects among $n$ people, possibly with the same person getting multiple objects

### Example
How many 4-letter words can be formed over the Latin alphabet (26 letters)?
$$
26^4
$$

How many 8-character passwords can be made using uppercase, lowercase, and digits?
$$
(26 + 26 + 10)^8 = 62^8
$$

> [!TIP]
> The Power Rule assumes **independence** between choices — each position in the sequence can be filled without constraints from the others.

![[Pasted image 20251001152716.png]]