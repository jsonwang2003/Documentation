> [!ABSTRACT]
> 
> While [[Suffix Arrays|Suffix Arrays]] provide $O(k \log n)$ search time, the **Burrows-Wheeler Transform (BWT)** allows us to reach the theoretical speed limit of string matching: **$O(k)$**. By rearranging a string into a reversible, compressed format and utilizing the **Last-to-First (L2F) mapping**, we can find all occurrences of a query string as fast as we can read the query itself, regardless of the genome's size.

---
## 1. The Burrows-Wheeler Transform

The BWT is a transformation that clumps similar characters together, making it ideal for compression (like **bzip2**) and genomic indexing.
### Construction (The "Matrix" Method)

Using **BANANA\$**:
1. **Generate Rotations:** Create all cyclic shifts of the string.
2. **Sort Rotations:** Sort them alphabetically (with `$` as the smallest character).
3. **Take the Last Column:** The last column of this sorted matrix is the BWT.

**BWT(BANANA\$) = ANNB\$AA**

---
## 2. The L2F (Last-to-First) Mapping

The "magic" of the BWT lies in the **L2F Property**: The $i$-th occurrence of a character $x$ in the **Last** column (the BWT) corresponds to the same character in the original string as the $i$-th occurrence of $x$ in the **First** column (the sorted characters).

### The Reconstruction Logic

Because each row is a rotation, the character in the **Last** column always precedes the character in the **First** column in the original string. By jumping from Last to First repeatedly, we can walk backward through the entire genome.

---
## 3. Backward Search: $O(k)$ Pattern Matching

Instead of binary search (which narrows a range in a sorted list), the **FM-Index** (Full-text index in Minute space) uses the BWT to perform **Backward Search**. We start with the _last_ character of the query and work toward the _first_.
### The Algorithm

1. Start with a range `[top, bottom]` covering the entire table.
2. For each character $c$ in the query (moving **right to left**):
    - Find the range of $c$ in the **Last** column within the current `[top, bottom]`.
    - Use the **L2F mapping** to update `top` and `bottom` to the corresponding positions in the **First** column.
3. If the range becomes empty, the pattern does not exist. Otherwise, the final range tells you exactly how many times the pattern occurs.

---
## 4. Why This is the "Gold Standard"

The BWT-based search is remarkably efficient for two reasons:
- **Time:** The search takes $O(k)$ time. Since we must read the $k$ characters of the query anyway, this is the best possible complexity. It does not depend on the length of the genome ($n$).
- **Space:** The BWT can be compressed using **Run-Length Encoding (RLE)**. In genomics, where large stretches of DNA are repetitive, the index can often be smaller than the original genome!

---
## 5. Comparison of Read Mapping Strategies

| **Strategy**               | **Preprocessing** | **Search Time**         | **Notes**                                                  |
| -------------------------- | ----------------- | ----------------------- | ---------------------------------------------------------- |
| [[Aho-Corasick Automaton]] | $O(m)$            | $O(n + \text{matches})$ | Preprocesses the reads.                                    |
| [[Suffix Arrays]]          | $O(n)$            | $O(k \log n)$           | Uses binary search.                                        |
| **BWT / FM-Index**         | $O(n)$            | **$O(k)$**              | Industry standard for modern aligners (e.g., BWA, Bowtie). |
