> [!ABSTRACT] 
> **Huffman Coding** is a landmark algorithm in information theory that provides the most efficient way to represent data using **variable-length prefix codes**. By assigning shorter bit-sequences to frequent symbols and longer ones to rare symbols, it approaches the theoretical limit of compression defined by **Shannon Entropy**.
> 
> More information on **Huffman Code** [[Huffman Code|here]]

---
## 1. The Necessity of Prefix Codes

To decode a message without ambiguity, a code must be a **prefix code**: no code for one symbol can be the prefix of a code for another.

- **Ambiguous (Non-Prefix):** A=`0`, C=`1`, G=`10`, T=`11`.
    - The string `110` could be `CCA`, `CG`, or `TA`. This causes decoding failure.
- **Unambiguous (Prefix):** A=`0`, C=`10`, G=`110`, T=`111`.
    - The string `110` can only be `G`.

---
## 2. Tree Construction: The Bottom-Up Approach

Huffman’s algorithm builds the coding tree by prioritizing symbols with the lowest frequencies.
1. **Count:** Find the frequency of every symbol in the input.
2. **Forest:** Create a "leaf node" for each symbol.
3. **Merge:**
    - Pick the **two nodes** with the lowest frequencies.
    - Create a new parent node with a frequency equal to the sum of the two children.
    - Assign `1` to the left branch and `0` to the right.
4. **Repeat:** Continue merging until only one root node remains.

---
## 3. Encoding and Decoding

### Message Encoding (Root to Leaf)

To encode a symbol, follow the path from the root to its leaf.

> [!Tip] Implementation Tip: 
> In code, we often traverse from the **leaf up to the root** and use a **Stack** to reverse the bits, ensuring the output follows the correct root-to-leaf order.

### Message Decoding (Bit by Bit)

1. Start at the **root** of the Huffman Tree.
2. Read a bit: if `1`, move to the left child; if `0`, move to the right child.
3. If you reach a **leaf**, output the symbol stored there and return to the root.
4. Repeat until all bits are processed.

---
## 4. Huffman vs. The Real World

### The Header Requirement

Since Huffman trees are unique to each file's specific frequencies, the **tree structure** (or the frequency data) must be stored in the file header. Without this, the recipient cannot reconstruct the tree to decode the bits.

### Lossless vs. Lossy Compression

- **Huffman is Lossless:** The decompressed file is a perfect, bit-for-bit replica of the original.
- **Lossy Compression:** Formats like **MP3** or **JPEG** achieve much higher compression by discarding "unnoticeable" data (e.g., audio frequencies the human ear can't perceive).

|Type|Examples|Best For|
|---|---|---|
|**Lossless**|Huffman, ZIP, PNG, FLAC|Text, Source Code, Medical Data|
|**Lossy**|MP3, JPEG, WebP|Music, Photos, Streaming Video|

---
### Adaptive Huffman (A Bio-Inspiration)

As noted in your genome example, symbol frequencies can change within a single file. Modern compressors often "reset" or adapt the Huffman tree for different segments of a file (e.g., GC-rich vs. AT-rich regions) to achieve even tighter compression.