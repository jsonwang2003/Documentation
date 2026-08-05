> [!ABSTRACT]
> 
> Variable Length CBC is an encoding strategy where different symbols in an alphabet are represented by bitstrings of different lengths. The goal is to optimize space by assigning shorter codes to frequent symbols and longer codes to rare ones, reducing the average number of bits per character.

---
## The Core Concept
While [[Fixed Length Character-By-Character Encoding For Strings (Fixed Length CBC)]] is efficient when symbols appear with equal probability, it is wasteful when certain characters (like 'E' or 'A') appear much more often than others (like 'Z' or 'Q').

Variable Length CBC adjusts the "cost" of each character based on its frequency. However, this flexibility introduces a major challenge: **Ambiguity in Decoding**.

---
## The Problem of Ambiguity
In fixed-length schemes, the decoder knows exactly how many bits to read for each character (e.g., every 3 bits). In variable-length schemes, the boundaries between characters are not naturally defined.
### Example: Encoding "DAD"
Given this scheme:

|**Letter**|**Encoding**|
|---|---|
|**A**|0|
|**B**|1|
|**C**|11|
|**D**|10|

**Encoding**: $D(10) + A(0) + D(10) \implies \mathbf{10010}$

The Decoding Dilemma:
The string 10010 could be interpreted in multiple ways:
- `10` | `0` | `10` $\to$ **DAD**
- `1` | `0` | `0` | `10` $\to$ **BAAD**
- `1` | `0` | `0` | `1` | `0` $\to$ **BAABA**
- `10` | `0` | `1` | `0` $\to$ **DABA**

---
## Solutions to Ambiguity
### 1. Comma-Separated Codes
Similar to Morse Code (which uses pauses), a specific bit pattern or "comma" can be inserted between characters to act as a delimiter.
- **Downside**: This adds significant overhead, potentially negating the space savings of the variable-length scheme.
### 2. Prefix-Free Codes (Comma-Free)
A more efficient solution is to design the code so that **no codeword is a prefix of any other codeword**.
- If 'A' is `0`, no other character can start with `0`.
- This allows the decoder to recognize a character the moment its unique bit pattern is completed, without needing a separator.

---
## Why Fixed Length Avoids This
Fixed-length encoding is implicitly prefix-free because all strings are of length $n$. Since all codes stop at the same bit count, no code can be a "start" to a longer code. The decoder simply "chops" the bitstream into uniform blocks of size $\lceil \log_2 |Data| \rceil$.

---
## Implementation Strategies
- **[[Huffman Code]]**: The most famous implementation of a prefix-free, variable-length code. It uses a greedy algorithm to build an optimal binary tree based on character frequencies.

---
## Related Notes
- [[Lossless Encoding]] – The requirement for one-to-one mapping in reconstruction.
- [[Strings as Integers]] – Reaching the theoretical minimum for fixed-length strings.
- [[Binary Tree]] – The structural foundation for prefix-free decoding.