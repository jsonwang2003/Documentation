> [!ABSTRACT]
> 
> Fixed Length CBC is the simplest method for mapping an alphabet to binary. By assigning every character a bitstring of the exact same length, we ensure that decoding is a simple matter of "chopping" the binary stream into uniform blocks.

---
## The Core Logic
In this scheme, we treat each character as an independent unit.
1. **Count** the number of unique characters in the alphabet ($x$).
2. **Calculate** the uniform bit length ($k$) needed to provide a unique binary address for each character.
3. **Map** each character to a unique binary sequence of length $k$.
### The Minimum Bits Formula
If an alphabet has $x$ characters, the minimum number of bits per character is:

$$
k = \lceil \log_2 x \rceil
$$

> [!TIP]
> 
> This formula ensures we have enough "slots" ($2^k$) to cover all $x$ characters. If $2^k > x$, some bitstrings will simply remain unused (e.g., in an alphabet of 6, the strings 110 and 111 are "wasted" if using 3-bit encoding).

---
## Examples & Deep Dive
### 1. The $\{A...F\}$ Alphabet
For an alphabet of 6 characters: $\lceil \log_2 6 \rceil = 3$ bits per character.

|**Letter**|**Binary**|
|---|---|
|**A**|`000`|
|**B**|`001`|
|**C**|`010`|
|**D**|`011`|
|**E**|`100`|
|**F**|`101`|

**Encoding "BADD":**
- $B(001) + A(000) + D(011) + D(011) = \mathbf{001000011011}$ (Total 12 bits)
### 2. Industry Standard: [[ASCII]]
ASCII is the most famous Fixed Length CBC. It uses an 8-bit (1 byte) fixed length to represent 256 possible characters (including uppercase, lowercase, numbers, and symbols).
- Even a simple character like `!`, which could theoretically be represented with fewer bits in a tiny alphabet, still takes up exactly 8 bits in ASCII to maintain the fixed-length property.

---
## The Efficiency Trade-off
### The "Waste" of Independence
Fixed Length CBC is often **sub-optimal** because it treats every character as a separate entity rather than looking at the string as a whole.

For example, a 4-letter string using the $\{A...F\}$ alphabet has $6^4 = 1296$ possible combinations.
- **Fixed Length CBC**: Always uses $4 \times 3 = \mathbf{12}$ bits.
- **Theoretical Optimum**: $\lceil \log_2(6^4) \rceil = \mathbf{11}$ bits.

This "lost bit" occurs because *Fixed Length CBC* doesn't account for the mathematical relationships between positions in a string. To reach that 11-bit optimum, you would need to encode the entire string at once using [[Strings as Integers]].

---
## Pros and Cons

| **Strengths**                                                                                                | **Weaknesses**                                                                                                   |
| ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **Instant Decoding**: No need to look ahead; just read $k$ bits at a time.                                   | **Space Inefficiency**: Often uses more bits than the theoretical optimum.                                       |
| **Random Access**: You can jump to the $i$-th character easily by calculating the bit offset ($i \times k$). | **Uniformity Trap**: Uses the same space for frequent characters (like 'E') as it does for rare ones (like 'Z'). |
| **Robustness**: A single bit error only affects one character, not the entire subsequent string.             |                                                                                                                  |

---
## Related Notes
- [[Variable Length Character-By-Character Encoding for Strings (Variable Length CBC)]] — How we save space by making frequent characters "shorter."
- [[Huffman Code]] — An algorithm for finding the most efficient variable-length code.
- [[Lossless Encoding]] — Why we can't always compress data infinitely.