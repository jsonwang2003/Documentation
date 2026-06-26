---
tags:
  - Anagram
  - Quotient-rule
---


> [!INFO]
> An **anagram** is a rearrangement of a multiset of characters.  
> When characters repeat, we divide by the factorial of their frequencies to avoid overcounting identical arrangements.

---

### General Formula

If a word has $n$ total letters, and some letters repeat with frequencies $f_1, f_2, \dots, f_k$, then the number of distinct anagrams is:

$$
\frac{n!}{f_1! \cdot f_2! \cdot \dots \cdot f_k!}
$$

---

### Examples

#### 1. Letters {E, E, T, S}

- Total permutations of labeled characters: $4!$
- Repetition of E counted $2!$ times

$$
\frac{4!}{2!} = 12
$$

#### 2. Letters {O, O, F, F, N}

- Total permutations: $5!$
- Two letters (O and F) each repeated twice

$$
\frac{5!}{2! \cdot 2!} = 30
$$

#### 3. Word: REASSESS

- Letters: R, E, A, S, S, E, S, S
- Frequencies: R (1), E (2), A (1), S (4)

$$
\frac{8!}{1! \cdot 1! \cdot 2! \cdot 4!} = 8 \cdot 7 \cdot 5 \cdot 3 = 840
$$

> [!NOTE]
> This is a classic use of the [[CountingTechniques/QuotientRule]].