> [!ABSTRACT]
> 
> Lossless encoding is a method of data representation that allows the original data to be perfectly reconstructed from the compressed data. To achieve this, the encoding function must be one-to-one (injective), ensuring every unique piece of data maps to a unique binary string.

![[Pasted image 20251120131748.png]]

---
## Efficiency in Encoding
When evaluating an encoding algorithm, we measure efficiency in two primary dimensions:
1. **Space**: The size of the encoded bitstring (compression ratio).
2. **Time**: The computational complexity required to encode and decode the information.

---
## The Mathematical Foundation
For an encoding to be lossless, it must function as a one-to-one mapping. This requirement dictates the minimum number of bits needed to represent a set of data.
### Minimum Bit Requirement
If we have a set of data $|Data|$, and we want to encode it into binary strings of length $n$:
1. The number of unique data items must be less than or equal to the total number of available binary strings:
    $$
    |Data| \leq 2^n
    $$
    
2. Solving for $n$ gives the theoretical lower bound:
    $$
    \boxed{n \geq \lceil \log_2 |Data| \rceil}
    $$
    

There will **always** exist an encoding of length $\lceil \log_2 |Data| \rceil$ bits for any finite data set.

---
## Encoding as a Function
An encoding of a set of Data is a map $E$ where the domain is the **Set of Data** and the codomain is a set of binary strings.
### Fixed Length Encoding
- **Codomain**: $\{0, 1\}^n$ (The finite set of binary strings of length $n$).
- **Mechanism**: Maps every data element to a string of exactly $n$ bits.
- **Usage**: Best for random access and simple hardware implementations (e.g., standard ASCII).
### Variable Length Encoding
- **Codomain**: $\{0, 1\}^*$ (The infinite set of all binary strings of all possible lengths).
- **Mechanism**: Maps data elements to strings of varying lengths.
- **Usage**: Optimized for efficiency based on complexity or frequency.
    - **More complex/rare data**: Requires more bits.
    - **Simpler/frequent data**: Requires fewer bits.

> [!NOTE]
> 
> Variable length encoding is essential for effective data compression, as it allows us to minimize the "average" length of the encoded data.

---
## Strategies for Encoding Strings
There are several ways to apply these functional mappings to strings of characters:
- [[Fixed Length Character-By-Character Encoding For Strings (Fixed Length CBC)]]: Maps each character to a fixed number of bits.
- [[Variable Length Character-By-Character Encoding for Strings (Variable Length CBC)]]: Maps each character to a variable number of bits (e.g., [[Huffman Code]]).
- [[Strings as Integers]]: Treats the entire string as a single numerical value to achieve the theoretical optimum of $\lceil \log_2 |Data| \rceil$.