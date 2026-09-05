> [!ABSTRACT]
> 
> Huffman coding is a variable-length, prefix-free encoding scheme. It achieves data compression by assigning shorter bitstrings to more frequent symbols and longer bitstrings to less frequent ones. It is mathematically proven to be an optimal character-by-character encoding method.

---
## The Core Concept
Unlike [[Fixed Length Character-By-Character Encoding For Strings (Fixed Length CBC)|Fixed Length CBC]], where every character is taxed the same amount of bits, Huffman coding is **adaptive**. It minimizes the total weighted path length of a binary tree, where weights represent character frequencies.
### Key Properties
- **Prefix-Free**: No codeword is a prefix of any other codeword (e.g., if 'A' is `01`, no other character can start with `01`). This makes the code **comma-free** or self-delimiting.
- **Greedy Approach**: The algorithm builds the tree from the "bottom up" by repeatedly merging the two least frequent nodes.

```pseudo
	\begin{algorithm}
	\caption{Huffman Coding}
	\begin{algorithmic}
	\Procedure{Huffman}{$C$: symbols $a_i$ with frequencies $w_i$, $i=1, \dots, n$}
		\State $F = $ Forst of $n$ rooted trees each consisting of the single vertex $a_i$ and assigned weighted $w_i$
		\While{$F$ is not a tree}
			\State Replace the rooted trees $T$ and $T'$ of least weights from $F$ with $w(T) \geq w(T')$ with a tree having a new root that has $T$ as its $left$ subtree and $T'$ as its $right$ subtree
			\State Label the new edge to $T$ with $0$
			\State Label the new edge to $T'$ with $1$
			\State Assign $w(T) + w(T')$ as the weight of the new tree
        \EndWhile
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---
## Building a Huffman Tree: Step-by-Step
### 1. Initialize and Sort
List all characters and their frequencies. Sort them in **decreasing order**. If frequencies are tied, use alphabetical order as a tie-breaker.

![[Pasted image 20251120191023.png]]
### 2. The Merge Loop
Repeat the following until only one tree (the root) remains:
1. Take the **two nodes with the lowest frequencies**.
2. Combine them into a new internal node.
3. The frequency of the new node is the **sum** of the two children.
4. Re-insert this new node into the list and **re-sort**.

![[Pasted image 20251120191105.png]]

![[Pasted image 20251120191119.png]]

![[Pasted image 20251120192115.png]]

![[Pasted image 20251120192130.png]]

![[Pasted image 20251120192141.png]]

![[Pasted image 20251120192152.png]]

![[Pasted image 20251120192208.png]]

![[Pasted image 20251120192218.png]]

![[Pasted image 20251120192230.png]]

![[Pasted image 20251120192247.png]]

### 3. Decorate the Tree
Assign a binary value to each edge:
- Left branch = `0`
- Right branch = `1`

![[Pasted image 20251120192304.png]]

---
## Practical Example Analysis
Given the dataset:

| **Character** | **Frequency** |
| ------------- | ------------- |
| A             | 6             |
| B             | 5             |
| C             | 4             |
| D             | 4             |
| E             | 2             |
| F             | 2             |
| G             | 1             |

### Resulting Codewords

|**Letter**|**Frequency**|**Huffman Code**|**Length (bits)**|
|---|---|---|---|
|**A**|6|`01`|2|
|**B**|5|`10`|2|
|**C**|4|`000`|3|
|**D**|4|`001`|3|
|**E**|2|`111`|3|
|**F**|2|`1100`|4|
|**G**|1|`1101`|4|

---
## Efficiency Comparison: Fixed vs. Variable
For a message containing the 24 characters above:
### 1. Fixed-Length Analysis
- **Alphabet Size**: 7 unique characters.
- **Bits per Char**: $\lceil \log_2 7 \rceil = 3$ bits.
- **Total Size**: $3 \times 24 = \mathbf{72 \text{ bits}}$.
### 2. Huffman Analysis
The size is calculated by $\sum (\text{frequency}_i \times \text{length}_i)$

$$
(6 \times 2) + (5 \times 2) + (4 \times 3) + (4 \times 3) + (2 \times 3) + (2 \times 4) + (1 \times 4) = \mathbf{64 \text{ bits}}
$$

**Total Savings**: $\approx 11\%$ compression over fixed-length.

---
## Decoding Logic
Decoding is efficient because of the **Prefix-Free** property.
1. Start at the **root**.
2. Read the bitstream:
    - If `0`, move Left.
    - If `1`, move Right.
3. When you reach a **leaf**, output that character.
4. **Immediately jump back to the root** for the very next bit.

---
## Related Notes
- [[Lossless Encoding]] – Why prefix-free codes are necessary for perfect reconstruction.
- [[Fixed Length Character-By-Character Encoding For Strings (Fixed Length CBC)]] – The non-adaptive alternative.
- [[Variable Length Character-By-Character Encoding for Strings (Variable Length CBC)]] – General theory of non-uniform bit lengths.
- [[Data Structure of Huffman Code]] – Data Structure for Huffman Coding