---
description: "A compact prefix-tree structure compressing database transaction logs to discover frequent itemsets without candidate generation."
aliases:
  - Frequent Pattern Tree
  - FP-Tree
  - FP-Growth Tree
tags:
  - data-structures
  - trees
  - data-mining
  - association-rules
---
> [!abstract] Abstract 
> A Frequent Pattern Tree (FP-Tree) is a specialized prefix-tree structure used in association rule mining and data analysis. Deployed within the **FP-Growth (Frequent Pattern Growth)** algorithm, it compresses large transaction databases into a dense tree layout without explicitly generating costly candidate itemsets (unlike the traditional Apriori algorithm).
> 
> - **Category:** Prefix-Based Mining Tree
> - **Primary Objective:** High-throughput frequent itemset mining across transaction streams.
> - **Core Advantage:** Eliminates repeat database scans and candidate pair generation loops.

---

# Core Architectural Components

An FP-Tree compresses transaction data by sharing common prefix item paths while maintaining structural links for rapid pattern mining:

1.  **The FP-Tree Root & Nodes:**
    *   `item_id`: The discrete item key stored at the node.
    *   `count`: An integer tally recording how many transactions share this path prefix.
    *   `parent`: A pointer reference tracing back toward the root ancestor.
    *   `node_link`: A dynamic pointer connecting to the next node in the tree holding the *same* `item_id`.
2.  **The Item Header Table:**
    A auxiliary table tracking all frequent items sorted in **descending order of global support frequency**. Each header entry contains:
    *   The item identifier.
    *   The total global support count across the dataset.
    *   A `head_ptr` referencing the first node instance in the FP-Tree, forming a linked chain across all matching item nodes via `node_link` pointers.

---

# Construction Pipeline

To construct an FP-Tree from a raw transaction database, the engine runs a two-pass scan:

### Pass 1: Global Frequency Counting & Filtering
Scan the database to compute the support frequency of every item. Filter out items whose frequency falls below a predefined minimum support threshold ($min\_sup$). Sort the remaining frequent items in **descending order of frequency**.

### Pass 2: Ordered Transaction Insertion
Scan the database a second time. For each transaction:
1.  Filter out non-frequent items.
2.  Reorder the remaining items according to the global descending frequency order established in Pass 1.
3.  Insert the ordered item list into the FP-Tree, sharing existing branch prefixes and incrementing node `count` counters whenever prefixes match.

```pseudo
	\begin{algorithm}
	\caption{FP-Tree Node Insertion Routine}
	\begin{algorithmic}
		\Procedure{InsertFPTree}{items, currNode, headerTable}
			\If{items is empty}
				\Return
			\EndIf
			\State $p \gets \text{First item in } items$
			\State $rest \gets \text{Remaining items in } items$
			\If{currNode has child $c$ matching $p$}
				\State $c.\text{count} \gets c.\text{count} + 1$
				\State \Call{InsertFPTree}{$rest, c, headerTable$}
			\Else
				\State $newNode \gets \text{CreateNode}(p, \text{count}=1, \text{parent}=currNode)$
				\State \Call{AttachChild}{currNode, newNode}
				\State \Call{UpdateHeaderLink}{$headerTable[p], newNode$}
				\State \Call{InsertFPTree}{$rest, newNode, headerTable$}
			\EndIf
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# The FP-Growth Mining Strategy

Once the FP-Tree is constructed, frequent itemsets are extracted recursively without scanning the original raw database again:

1.  **Conditional Pattern Bases:** For each item in the Header Table (processed from lowest frequency to highest), traverse its `node_link` chain to extract all prefix paths leading from the root to those nodes.
2.  **Conditional FP-Trees:** Build a localized "conditional FP-Tree" using those prefix paths, weighted by the path counts.
3.  **Recursive Mining:** Recursively mine the conditional tree to output all frequent item combinations containing the target item.

---

# Architectural Trade-Offs & Performance

| Parameter Metric | Apriori Algorithm Baseline | FP-Tree / FP-Growth Mining |
|---|---|---|
| **Database Scans** | $k+1$ scans (where $k$ is max pattern length) | **Exactly 2 scans** regardless of pattern length |
| **Candidate Generation** | Explicitly generates millions of candidate pairs | **Zero candidate pair generation** |
| **Search Mechanism** | Combinatorial join passes | Divide-and-conquer prefix tree walks |
| **Memory Footprint** | Low initial footprint | High tree memory overhead for sparse datasets |

> [!tip] Prefix Compression Benefits
> When transactions contain highly repeated items, the FP-Tree achieves immense structural compression, allowing massive transaction datasets to fit entirely within high-speed RAM.

---

# Related Notes

- [[Tree Structures/Binary Tree|Binary Tree]]
- [[Lexicon/Multiway Trie Implementation|Multiway Trie]]
- [[Hashing/Count-Min Sketches|Count-Min Sketches]]