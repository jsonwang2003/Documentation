> [!ABSTRACT]
> 
> Encoding strings as integers is a method that treats an entire string of length $n$ as a single numerical value rather than a sequence of independent characters. This approach allows us to reach the theoretical minimum bit-length for encoding, overcoming the inefficiencies found in [[Fixed Length Character-By-Character Encoding For Strings (Fixed Length CBC)]].

---
## Theoretical Minimum
When encoding $n$-letter strings over an alphabet $X$, the total number of possible strings is $|X|^n$. To map each unique string to a unique binary sequence, we need:

$$\text{Minimum Bits} = \lceil \log_2(|X|^n) \rceil$$

This method is more space-efficient because it utilizes the "gaps" that character-by-character encoding leaves behind when the alphabet size is not a perfect power of 2.

---
## The Encoding Procedure
### Step 1: Alphabet Mapping
Establish an arbitrary ordering on the alphabet $X$. Assign each character a value from $0$ to $|X| - 1$.
- Example: If $X = \{A, B, C, D, E, F\}$, then $A=0, B=1, C=2, D=3, E=4, F=5$.
### Step 2: Base Conversion (String to Integer)
Treat the string $s$ as a number in base-$|X|$. Convert the string into a single integer using the assigned values.
- A string $s_1s_2s_3s_4$ in base-$|X|$ is calculated as:
    $$s_1 \cdot |X|^3 + s_2 \cdot |X|^2 + s_3 \cdot |X|^1 + s_4 \cdot |X|^0$$
    
### Step 3: Binary Conversion
Convert the resulting base-$|X|$ integer into a base-2 (binary) string. Ensure the output is a fixed-width string of length $\lceil \log_2 |X|^n \rceil$ by adding leading zeros if necessary.

---
## Worked Example: 4-Letter Strings over $\{A, B, C, D, E, F\}$

**Parameters:**
- Alphabet size $|X| = 6$
- String length $n = 4$
- Total possible strings: $6^4 = 1296$
- Required bits: $\lceil \log_2 1296 \rceil = \boxed{11 \text{ bits}}$
### Encoding "BEAD"
1. **Values**: $B=1, E=4, A=0, D=3$
2. Base-6 to Integer:
    $$(1 \cdot 6^3) + (4 \cdot 6^2) + (0 \cdot 6^1) + (3 \cdot 6^0)$$
    $$216 + 144 + 0 + 3 = \mathbf{363_{10}}$$
    
3. Integer to Binary (11-bit fixed width):
    $$363_{10} = (101101011)_2$$
    Padding to 11 bits $\implies \mathbf{00101101011}$
    
### Encoding "FFFF"
1. **Values**: $F=5, F=5, F=5, F=5$
2. Base-6 to Integer:
    $$(5 \cdot 6^3) + (5 \cdot 6^2) + (5 \cdot 6^1) + (5 \cdot 6^0) = \mathbf{1295_{10}}$$
    
3. Integer to Binary:
    $$1295_{10} = \mathbf{10100001111}$$
    

---
## Comparison of Efficiency
Using the alphabet $\{A, B, C, D, E, F\}$ for a 4-letter string:

|**Encoding Method**|**Calculation**|**Total Bits**|
|---|---|---|
|**Fixed Length CBC**|$4 \text{ chars} \cdot 3 \text{ bits/char}$|12 bits|
|**Strings as Integers**|$\lceil \log_2 6^4 \rceil$|**11 bits**|

**Conclusion**: By treating the string as a single integer, we save 1 bit per 4 characters in this specific alphabet. Over very long strings, this efficiency gain scales significantly.

---
## Related Notes
- [[Lossless Encoding]] – The requirements for one-to-one mapping.
- [[Fixed Length Character-By-Character Encoding For Strings (Fixed Length CBC)]] – Comparison of simplicity vs. space efficiency.
- [[Variable Length Character-By-Character Encoding for Strings (Variable Length CBC)]] – Another approach to compression based on frequency.