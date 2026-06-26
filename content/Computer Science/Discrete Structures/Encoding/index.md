---
title: Encoding
---
> [!ABSTRACT]
> 
> Encoding is the process of converting data from one form to another. In discrete algorithms, we focus on Lossless Encoding—representing strings, characters, and integers in ways that minimize space without losing any original information.

---
## Knowledge Map
### Fundamentals
- **[[Strings as Integers]]**: Understanding how text can be mapped to numerical values using base-$k$ representations, forming the basis for many hashing and storage techniques.
- **[[Lossless Encoding]]**: The overarching theory that data must be perfectly re-constructible. This introduces the concept of **Prefix Codes** (where no code is a prefix of another).
### Character-By-Character (CBC) Encoding
There are two primary ways to map characters to binary strings:
1. **[[Fixed Length Character-By-Character Encoding For Strings (Fixed Length CBC)]]**
    - **Concept**: Every character uses the same number of bits (e.g., ASCII uses 8 bits).
    - **Efficiency**: Easy to index but wasteful if some characters appear more frequently than others.
2. **[[Variable Length Character-By-Character Encoding for Strings (Variable Length CBC)]]**
    - **Concept**: Frequent characters get shorter bit-strings; rare characters get longer ones.
    - **Requirement**: Requires careful design to remain "uniquely decodable."
### Optimization
- **[[Huffman Code]]**
    - **The Algorithm**: A greedy strategy that builds an optimal prefix tree based on character frequencies.
    - **Complexity**: Uses a priority queue to achieve $O(k \log k)$ where $k$ is the alphabet size.
    - **Result**: Proven to provide the minimum possible bit-length for a given frequency distribution.

---
## Comparison Table

|**Method**|**Bit Length**|**Best Used For...**|
|---|---|---|
|**Fixed Length**|$\lceil \log_2(\text{alphabet size}) \rceil$|Simple systems, random access.|
|**Variable Length**|Varies|Reducing average file size.|
|**Huffman Coding**|**Optimal**|Maximum compression based on frequency.|

---
## Related Toolkits
- [[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Divide and Conquer/index|Divide and Conquer]]: Often compared with the "Greedy" approach used in Huffman trees.
- [[Asymptotic Notation]]: Analyzing the runtime of building the encoding trees.