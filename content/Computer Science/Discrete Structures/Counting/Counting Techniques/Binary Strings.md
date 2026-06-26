---
tags:
  - Counting
  - Binary
  - Combinatorics
  - Power-Rule
  - Complement
  - Inclusion-Exclusion
---
> [!INFO]
> A sequence composed solely of the characters `0` and `1`.  
> These strings are fundamental in computer science and digital systems—used to represent binary data, machine instructions, and logic states.

### How Many Binary Strings of Length $n$?

$$
|\{0, 1\} \times \{0, 1\} \times \dots \times \{0, 1\}| = |\{0, 1\}|^n = 2^n
$$

> [!IMPORTANT]
> In general, the number of strings of length $n$ over an alphabet with $X$ characters is:

$$
X^n
$$

- $X$: number of choices per position  
- $n$: number of positions

See also: [[Power Rule]]

---
## Examples

#### 1. Latin Alphabet Words

How many 4-letter words can be formed over the Latin alphabet (26 letters)?

$$
26^4
$$

#### 2. Passwords with Mixed Characters

How many 8-character passwords can be made using uppercase, lowercase, and digits?

$$
(26 + 26 + 10)^8 = 62^8
$$

#### 3. Passwords with Disjoint Character Sets

How many 8-character passwords contain only uppercase, only lowercase, or only digits?

$$
26^8 + 26^8 + 10^8
$$

> [!NOTE]
> This is a direct application of the [[Sum Rule]].

---

### Complement Method: “At Least One” Conditions

#### 4. At Least One Uppercase Letter

How many 8-character passwords contain at least one uppercase letter?

Let:
- Total passwords: $62^8$
- Passwords with no uppercase (only lowercase + digits): $36^8$

Then:

$$
\text{At least one uppercase} = 62^8 - 36^8
$$

Alternate form:

$$
26 \cdot 62^7
$$

> [!TIP]
> This is a classic use of the complement principle. See also: [[Inclusion Exclusion]]

---

### Complement Method: “At Least One 0”

#### 5. 4-Digit Strings with at Least One 0

How many 4-digit strings (digits 0–9) contain at least one 0?

- Total: $10^4$
- No 0s (digits 1–9 only): $9^4$

So:

$$
10^4 - 9^4
$$

---

### Exact Count Methods

#### 6. Exactly $k$ Zeros in a 4-Digit String

Breakdown:

$$
(4 \cdot 9^3) + (6 \cdot 9^2) + (4 \cdot 9) + 1
$$

#### 7. First Occurrence of 0

Alternative breakdown:

$$
10^3 + (9 \cdot 10^2) + (9^2 \cdot 10) + 9^3
$$

---

### Distribution Interpretation

#### 8. Distributing 10 Distinct Candies to 4 Children

If each candy independently goes to one of 4 children:

$$
4^{10}
$$

If each child gets exactly one candy and order matters:

$$
10 \cdot 9 \cdot 8 \cdot 7 = P(10, 4)
$$

> [!NOTE]
> See [[Power Rule]] and [[rPermutations]] for context.