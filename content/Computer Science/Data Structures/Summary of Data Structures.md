## [[Array Lists|Array List]]
### Summary Description

An ArrayList is an Abstract Data Type (ADT) implemented using a Dynamic Array. It acts as a wrapper around a standard fixed-size array, providing a mechanism for automatic resizing when the capacity is reached.

#### Key Features

- Random Access: Leveraging the underlying array's properties, access to any element via its index is performed in $O(1)$ time.
    
- Contiguity: Elements are stored in contiguous memory locations. There are no "gaps" or empty slots between the first and last element.
    
- Resizing Logic: When the array becomes full, the container typically allocates a new array (usually double the current size), copies all elements, and deallocates the old array.
    

### Complexity Analysis: Unsorted ArrayList

#### Performance Metrics

- Find: Best Case $O(1)$, Average Case $O(n)$, Worst Case $O(n)$
    
- Insert: Best Case $O(1)$, Average Case $O(n)$, Worst Case $O(n)$
    
- Remove: Best Case $O(1)$, Average Case $O(n)$, Worst Case $O(n)$
    

#### Mathematical Derivations

- Average Find:
    
    $$\frac{1+2+...+n}{n} = \frac{\sum_{i=1}^{n} i}{n} = \frac{n(n+1)}{2n} = \frac{n+1}{2} \approx O(n)$$
    
- Average Insert:
    
    $$\frac{n+(n-1)+...+0}{n} = \frac{\sum_{i=0}^{n} i}{n} = \frac{n(n+1)}{2n} = \frac{n+1}{2} \approx O(n)$$
    
- Average Remove:
    
    $$\frac{(n-1)+(n-2)+...+0}{n} = \frac{\sum_{i=0}^{n-1} i}{n} = \frac{n(n-1)}{2n} = \frac{n-1}{2} \approx O(n)$$
    

### Complexity Analysis: Sorted ArrayList

#### Performance Metrics

- Find: Best Case $O(1)$, Average Case $O(\log n)$, Worst Case $O(\log n)$
    
- Insert: Best Case $O(1)$, Average Case $O(n)$, Worst Case $O(n)$
    
- Remove: Best Case $O(1)$, Average Case $O(n)$, Worst Case $O(n)$
    

#### Key Differences

- Binary Search: Because the data is sorted and indexed, we can discard half the search space in each step, resulting in $O(\log n)$ performance for Find.
    
- Insertion/Removal: Even though we can find the target index in $O(\log n)$, we must still shift elements to maintain contiguity, keeping the operation at $O(n)$.
    

### Space Complexity

The space complexity for both sorted and unsorted ArrayLists is $O(n)$.

#### Memory Management

- Utilization: The actual memory allocated ($m$) is always greater than or equal to the number of elements ($n$).
    
- Resizing Bounds: In a typical doubling implementation, the array is at its most efficient just before resizing (where $m = n$) and its least efficient just after resizing (where $m = 2n$). In both cases, the relationship is linear.
    

### Summary Comparison

#### Comparison Table

|**Feature**|**Unsorted ArrayList**|**Sorted ArrayList**|
|---|---|---|
|Primary Advantage|Fastest Insertions (at end)|Fastest Search (Binary Search)|
|Search Method|Linear Search|Binary Search|
|Index Access|$O(1)$|$O(1)$|
|Contiguous?|Yes|Yes|


---
## [[Computer Science/Data Structures/Introductory Data Structures/Linked List|Linked List]]

### Summary Description

Linked Lists are linear data structures composed of nodes connected to one another via pointers. Unlike Array Lists, Linked Lists do not store elements in contiguous memory locations.

#### Structure and Access

- Global Pointers: We typically maintain access to two pointers: `head` (pointing to the first node) and `tail` (pointing to the last node).
    
- Singly-Linked List: Each node maintains a single `next` pointer that points to the subsequent node in the list.
    
- Doubly-Linked List: Each node maintains two pointers: a `next` pointer and a `previous` pointer, allowing for bidirectional traversal.
    
- Sequential Access: We do not have random access (indexing). To reach an inner node, we must traverse the list node-by-node starting from the `head` or `tail`.
    

#### Modification Logic

Once the target location is identified, the actual insertion or removal of a node is an $O(1)$ operation. This is achieved by simply redirecting the pointers of the neighboring nodes. Consequently, the total time complexity for these operations is dominated by the time required to find the target position.

### Complexity Analysis

#### Performance Metrics

- Find: Best Case $O(1)$, Average Case $O(n)$, Worst Case $O(n)$
    
- Insert: Best Case $O(1)$, Average Case $O(n)$, Worst Case $O(n)$
    
- Remove: Best Case $O(1)$, Average Case $O(n)$, Worst Case $O(n)$
    

#### Mathematical Derivations

- Average Find (Singly-Linked):
    
    $$\frac{1+2+...+n}{n} = \frac{\sum_{i=1}^{n} i}{n} = \frac{n(n+1)}{2n} = \frac{n+1}{2} \approx O(n)$$
    
- Average Find (Doubly-Linked): If we know the index and can choose to start from the `head` or the `tail` (whichever is closer), the average number of nodes visited is:
    
    $$\frac{2(\frac{n}{2}+1)}{2} = \frac{n}{4} + 1 \approx O(n)$$
    

### Space Complexity

The space complexity for a Linked List is $O(n)$.

#### Memory Allocation

- Node Overhead: Each of the $n$ nodes requires space for the stored data plus the overhead of 1 or 2 pointers (for singly- or doubly-linked, respectively).
    
- Per-Node Cost: Since each node uses $O(1)$ space for its pointers, the total space scales linearly with the number of elements.
    

### Summary Comparison

#### Comparison Table

|**Feature**|**Singly-Linked List**|**Doubly-Linked List**|
|---|---|---|
|Pointers per Node|1 (Next)|2 (Next & Previous)|
|Traversal|Forward Only|Bidirectional|
|Memory Overhead|Lower|Higher|
|Find (Worst Case)|$O(n)$|$O(n)$|

---
## [[Skip Lists]]

### Summary Description

Skip Lists are probabilistic data structures that augment a standard Linked List with multiple layers of forward pointers. This multi-level hierarchy allows the search process to "skip" over large sections of the list, effectively mimicking the efficiency of Binary Search in a linked structure.

#### Structure and Probabilistic Height

- Layers and Height: Each node contains a variable number of forward pointers, referred to as the node's height.
    
- Max Height: We typically define a maximum height $h$, where $h \ll n$.
    
- Coin Flipping: To maintain balance without complex reshuffling, node heights are determined randomly. A weighted coin with probability $p$ is flipped repeatedly; the height increases with each "heads" until the first "tails" or the maximum height is reached.
    
- Efficient Traversal: Searches begin at the highest level of the head node, moving forward as long as the next node's value is less than the target, and dropping down a level when the next value is too large.
    

#### Modification Logic

Similar to a Linked List, the cost of insertion or removal is dominated by the search time. Once the correct position is located, updating the pointers across the node's levels takes $O(h)$ time.

### Complexity Analysis

#### Performance Metrics

- Find: Best Case $O(1)$, Average Case $O(\log n)$, Worst Case $O(n)$
    
- Insert: Best Case $O(h)$, Average Case $O(\log n)$, Worst Case $O(n)$
    
- Remove: Best Case $O(h)$, Average Case $O(\log n)$, Worst Case $O(n)$
    

#### Analysis Details

- Worst Case: In the highly unlikely event that all nodes are assigned the same height (e.g., height 1), the structure degenerates into a standard Singly-Linked List with $O(n)$ performance.
    
- Average Case: Due to the probabilistic distribution of node heights, the expected search path length is logarithmic, providing $O(\log n)$ efficiency for finding, inserting, and removing elements.
    
- Best Case: The best case for a find occurs if the target is the very first element checked. For insertion and removal, even with a best-case find, we must still update pointers across $h$ levels.
    

### Space Complexity

The space complexity for a Skip List is generally discussed in terms of its expected and worst-case bounds.

- Expected Space: On average, the number of pointers is a constant multiple of $n$, resulting in an expected space complexity of $O(n)$.
    
- Worst-Case Space: In the worst-case scenario regarding height distribution, the space complexity can reach $O(n \log n)$ or $O(n \cdot h)$, though the probability of this occurring is extremely low given the coin-flip logic.
    

### Summary Comparison

#### Comparison Table

|**Feature**|**Linked List**|**Skip List**|
|---|---|---|
|Search Strategy|Linear|Multi-level "Skip"|
|Search Time (Average)|$O(n)$|$O(\log n)$|
|Implementation Type|Deterministic|Probabilistic|
|Structure|Single level|Logarithmic levels|

---
## [[Heap]]

### Summary Description

A Heap is a specialized complete binary tree that satisfies the **Heap Property**. Because it is a complete tree (all levels are fully filled except possibly the last, which is filled from left to right), it maintains a balanced height of $\log n$.

#### The Heap Property

- **Priority:** For any two nodes $A$ and $B$, if $A$ is the parent of $B$, then $A$ must have a higher or equal priority than $B$.
    
- **Min-Heap:** The parent's value is always less than or equal to its children's values. The root is the minimum element.
    
- **Max-Heap:** The parent's value is always greater than or equal to its children's values. The root is the maximum element.
    

#### Array List Implementation

Heaps are most efficiently implemented using an Array List (Dynamic Array). This provides better cache locality and eliminates the need for explicit pointers. In an array-based heap starting at index $i=0$:

- **Root:** Stored at index $0$.
    
- **Parent of node at $i$:** Located at index $\lfloor \frac{i-1}{2} \rfloor$.
    
- **Left Child of node at $i$:** Located at index $2i + 1$.
    
- **Right Child of node at $i$:** Located at index $2i + 2$.
    

### Complexity Analysis

#### Performance Metrics

- **Peek:** Best/Average/Worst Case $O(1)$. We simply access the element at index $0$.
    
- **Insert:** Best Case $O(1)$, Average/Worst Case $O(\log n)$.
    
- **Pop (Extract):** Best Case $O(1)$, Average/Worst Case $O(\log n)$.
    

#### Operational Logic

- **Insertion (Up-Heap/Bubble-up):** The new element is placed at the first available spot at the end of the array to maintain the "complete tree" property, then compared with its parent and swapped upwards until the heap property is restored.
    
- **Pop (Down-Heap/Trickle-down):** The root is removed, and the last element in the array is moved to the root position. This element is then swapped downwards with its highest-priority child until the heap property is restored.
    

### Space Complexity

The space complexity for a Heap is $O(n)$.

- **Storage efficiency:** Since it is implemented as an Array List, it follows the $O(n)$ space complexity of dynamic arrays. Because the tree is always complete, there are no null pointers or wasted internal nodes, making it very memory-efficient compared to a standard linked binary tree.
    

### Summary Comparison

#### Comparison Table

|**Feature**|**Heap**|**Sorted Array List**|
|---|---|---|
|Find Min/Max|$O(1)$|$O(1)$|
|Insert|$O(\log n)$|$O(n)$|
|Remove Min/Max|$O(\log n)$|$O(n)$|
|Structure|Complete Tree|Linear|

---
## Binary Search Tree (BSTs)

### Summary Description

A **Binary Search Tree (BST)** is a fundamental hierarchical data structure where each node follows a specific ordering rule: the value of any node is greater than all values in its left subtree and smaller than all values in its right subtree.

#### Specialized BST Variants

To solve the "degeneracy" problem (where a tree becomes a linear list), several variants exist:

- **[[Randomized Search Trees (Treap, RST)]]:** A combination of a BST and a Heap. Each node has a key (following BST property) and a random priority (following Heap property).
    
- **[[AVL Tree]]:** A strictly self-balancing BST where the heights of the left and right subtrees of any node differ by at most one.
    
- **[[Red-Black Tree]]:** A self-balancing BST that uses "colors" (red/black) and specific structural rules to maintain balance. It is less strict than AVL but more efficient for insertions and deletions.


### Complexity Analysis: [[Binary Search Tree (BSTs)|Regular BST]]

#### Performance Metrics

- **Find/Insert/Remove:**
    
    - **Worst Case:** $O(n)$ — Occurs if elements are inserted in sorted order, creating a "skewed" tree that resembles a Linked List.
        
    - **Average Case:** $O(\log n)$ — Assuming a random distribution of keys.
        
    - **Best Case:** $O(1)$ — If the operation happens at the root node.
        

#### Space Complexity

- **$O(n)$:** Each node stores data and three pointers (parent, left, right), requiring constant space per node.

### Complexity Analysis: [[Randomized Search Trees (Treap, RST)]]

#### Performance Metrics

- **Worst Case:** $O(n)$ — High probability of being logarithmic, but theoretically could degenerate if keys and random priorities align poorly.
    
- **Average Case:** $O(\log n)$ — Random priorities act as a safeguard against skewed input.
    

#### Space Complexity

- **$O(n)$:** Similar to BST, with the addition of a priority value stored in each node.
    


### Complexity Analysis: [[AVL Tree]]

#### Performance Metrics

- **Find/Insert/Remove (Worst Case):** $O(\log n)$ — The strict balance property guarantees logarithmic height.
    
- **Comparison:** Because AVL is strictly balanced, it is usually faster for **find** operations than Red-Black trees but requires more "rotations" during modifications.
    

#### Space Complexity

- **$O(n)$:** Nodes often store an additional height factor to manage balancing.
    

### Complexity Analysis: [[Red-Black Tree]]

#### Performance Metrics

- **Find/Insert/Remove (Worst Case):** $O(\log n)$ — Balance is maintained such that the longest path to a leaf is no more than twice as long as the shortest.
    
- **Comparison:** Red-Black trees are generally faster for **insert** and **remove** operations because they require fewer structural adjustments (rotations) compared to AVL trees.
    

#### Space Complexity

- **$O(n)$:** Nodes store a single bit of information for "color" (red or black).
    


### Summary Comparison Table

|**Feature**|**Regular BST**|**Randomized (Treap)**|**AVL Tree**|**Red-Black Tree**|
|---|---|---|---|---|
|**Balance Strategy**|None|Probabilistic (Priorities)|Strict Height Factor|Color-based Rules|
|**Search (Worst)**|$O(n)$|$O(n)$|$O(\log n)$|$O(\log n)$|
|**Modification (Worst)**|$O(n)$|$O(n)$|$O(\log n)$|$O(\log n)$|
|**Best Use Case**|Small/Random Data|General Purpose|Search-Heavy Apps|

---
## ## B-Tree and B+ Tree

### Summary Description

B-Trees and B+ Trees are "fat" balanced search trees designed to handle large amounts of data, particularly for systems that read and write large blocks of data (like databases and filesystems). They minimize disk I/O by having a high branching factor, which keeps the tree very flat.

#### B-Tree Fundamentals

- **Node Capacity:** Every internal node must have at least $b$ and at most $2b$ children.
    
- **Sorted Order:** Elements within a single node are kept in ascending order.
    
- **Child Placement:** Child pointers are positioned between elements, acting as dividers. For elements $i$ and $j$, the subtree between them contains all values $x$ such that $i < x < j$.
    
- **Perfect Balance:** All leaf nodes must reside on the same level.
    

#### B+ Tree Variant

A B+ Tree is an optimization where the internal nodes function only as a "road map" (containing search keys), while the actual data records are stored exclusively in the leaf nodes.

- **Internal vs. Leaf:** Internal nodes contain up to $M$ children and $M-1$ keys. Leaves contain the actual data records (up to $L$ records).
    
- **Search Keys:** Internal nodes store keys to direct the search. In many implementations, the smallest data record in a subtree is used as the search key in the parent.
    
- **Leaf Connectivity:** Leaves are often linked together in a list to allow for efficient range queries (e.g., "find all records between 10 and 50").
- 

### Complexity Analysis: B-Tree

#### Performance Metrics

- **Find:** Worst Case $O(\log n)$. The high branching factor makes this very fast.
    
- **Insert/Remove:** Worst Case $O(b \log n)$. This accounts for traversing the height ($O(\log n)$) and potentially shifting up to $O(b)$ elements within a node to maintain sorted order during a split or merge.
    
- **Best Case Find:** $O(1)$ if the target is in the root node.
    

#### Space Complexity

- **$O(n)$:** Nodes are required to be at least half-full (specifically at least $b-1$ keys). Even in the worst-case scenario where every node is at minimum occupancy, the space remains linear.
    

### Complexity Analysis: B+ Tree

#### Performance Metrics

- **Find:** Worst Case $O(\log n + \log L)$. You must travel to the leaf level ($O(\log n)$) and then find the record within the leaf node ($O(\log L)$).
    
- **Insert/Remove:** Worst Case $O(M \log n + L)$. This involves the search path plus the overhead of shifting $O(M)$ keys in internal nodes and $O(L)$ records in a leaf.
    
- **Best Case Find:** $O(\log n)$. Unlike a B-Tree, you **must** always travel to the leaf level to retrieve a record, even if the key is present in the root.
    

#### Space Complexity

- **$O(n)$:** Because nodes must be at least 50% full, the overhead for internal search keys and pointers remains proportional to the number of data records.
    

### Summary Comparison Table

| **Feature**          | **B-Tree**                          | **B+ Tree**                        |
| -------------------- | ----------------------------------- | ---------------------------------- |
| **Data Location**    | Any node (internal or leaf)         | **Only** leaf nodes                |
| **Search Path**      | Can end at any level                | Always ends at leaf level          |
| **Range Queries**    | Difficult (requires tree traversal) | Easy (leaves are linked)           |
| **Branching Factor** | Lower (data takes up node space)    | Higher (internal nodes are "slim") |

---
## Hash Table and Hash Map

### Summary Description

A **Hash Table** is a data structure that maps keys to indices in an array using a **hash function**. A **Hash Map** extends this concept by storing (key, value) pairs. The primary goal of a Hash Table is to achieve near-constant time performance for core operations.

#### Key Concepts

- **Hash Function:** Converts a key into an integer. For a function $h$ to be valid, if $key_1 = key_2$, then $h(key_1) = h(key_2)$. A "good" function minimizes collisions ($h(key_1) = h(key_2)$ when $key_1 \neq key_2$).
    
- **Load Factor ($\alpha$):** Defined as $\alpha = \frac{N}{M}$, where $N$ is the number of elements and $M$ is the table capacity. Generally, $\alpha$ should stay below $0.75$ to maintain performance.
    
- **Capacity:** Often chosen as a prime number to improve the distribution of keys and reduce collision patterns.
    

#### Collision Resolution Strategies

- **Open Addressing:** All elements are stored within the array itself.
    
    - **Linear Probing:** If a collision occurs, check the next available slot ($index + 1$).
        
    - **Double Hashing:** Uses a second hash function to determine the "step size" for probing ($index + i \cdot h_2(key)$).
        
    - **Random Hashing:** Uses a pseudorandom sequence seeded by the key to find available slots.
        
- **Closed Addressing (Separate Chaining):** Each array slot points to a separate data structure (like a Linked List or BST) that holds all keys hashing to that index.
    
- **Cuckoo Hashing:** Uses two hash functions and two possible slots per key. If a collision occurs, the new key "kicks out" the resident key, which then moves to its alternative slot.

### Complexity Analysis: Probing (Linear, Double, Random)

#### Performance Metrics

- **Find/Insert/Remove:**
    
    - **Worst Case:** $O(n)$ — Occurs if every key hashes to the same index or causes a massive cluster, requiring a probe through all elements.
        
    - **Average Case:** $O(1)$ — Provided the load factor is kept low and the hash function is uniform.
        
    - **Best Case:** $O(1)$ — The key is found at the initial hashed index with no collisions.
        

#### Space Complexity

- **$O(n)$:** The table size is maintained at a constant multiple of the number of elements.
    


### Complexity Analysis: Separate Chaining

#### Performance Metrics

- **Find/Remove:**
    
    - **Worst Case:** $O(n)$ — Occurs if all elements hash to the same bucket, turning the search into a linear scan of a Linked List.
        
    - **Average Case:** $O(1)$ — Assuming a uniform distribution where each bucket has a small, constant number of elements.
        
- **Insert:** $O(1)$ if prepending to a list; $O(n)$ in the worst case if checking for duplicates is required.
    

#### Space Complexity

- **$O(n)$:** Includes the array capacity plus the $O(1)$ space for each of the $n$ nodes in the chains.
    


### Complexity Analysis: Cuckoo Hashing

#### Performance Metrics

- **Find/Remove:**
    
    - **Worst Case:** $O(1)$ — A key is guaranteed to be in one of only two specific locations. This makes Cuckoo Hashing excellent for applications requiring predictable lookup times.
        
- **Insert:**
    
    - **Worst Case:** $O(n)$ — If a cycle of "kicking out" elements is detected, the table must be resized and rebuilt (rehashed).
        
    - **Average Case:** $O(1)$.
        

### Summary Comparison Table

| **Feature**           | **Linear Probing**             | **Separate Chaining**  | **Cuckoo Hashing**        |
| --------------------- | ------------------------------ | ---------------------- | ------------------------- |
| **Collision Logic**   | Find next empty slot           | Linked List per bucket | Kick out existing key     |
| **Find (Worst)**      | $O(n)$                         | $O(n)$                 | **$O(1)$**                |
| **Insert (Avg)**      | $O(1)$                         | $O(1)$                 | $O(1)$                    |
| **Space Efficiency**  | High (in-place)                | Lower (pointers)       | High (in-place)           |
| **Load Factor Limit** | Very sensitive ($< 0.5$ ideal) | Less sensitive         | Sensitive ($< 0.5$ ideal) |

---
## [[Multiway Trie]]

### Summary Description

A **Multiway Trie** is a specialized tree-based data structure used for retrieval, typically of strings over a defined alphabet $\Sigma$. Unlike other trees that store the entire key within a node, a Trie stores keys along the paths from the root, where each edge represents a single character.

#### Structure and Key Properties

- **Edge Labeling:** Each edge is labeled with a character from the alphabet $\Sigma$.
    
- **Node Degree:** Every node can have a maximum of $|\Sigma|$ children, representing each possible character in the alphabet.
    
- **Key Path:** A key is defined by the concatenation of edge labels on the path from the root to a "word node" (a node marked as the end of a valid string).
    
- **Parameters:** We use $k$ to represent the length of the longest key and $n$ for the total number of elements stored.
    

### Complexity Analysis

#### Performance Metrics

- **Find:** Worst Case $O(k)$. The algorithm traverses the tree character by character. Since each step involves looking up a pointer in an array of size $|\Sigma|$, each character is processed in $O(1)$ time relative to $n$.
    
- **Insert:** Worst Case $O(k)$. Similar to find, the algorithm follows the path of the string and creates new nodes/edges ($O(1)$ each) for any characters not already present.
    
- **Remove:** Worst Case $O(k)$. After finding the "word node," the "is-word" marker is removed. Depending on the implementation, the algorithm may also prune the branch upwards if the nodes are no longer needed.
    

#### Average and Best Case

- **Average Case:** $O(k)$. While the formal complexity is $O(k)$, the practical speed depends on the distribution of key lengths. If keys are distributed uniformly, the expected time scales with the average length of the strings.
    
- **Best Case:** $O(1)$. For `Find` and `Remove`, if the first character of the query is not a child of the root, the operation terminates immediately.
    

### Space Complexity

The space complexity of a Multiway Trie is often its primary drawback, as it prioritizes speed over memory efficiency.

- **Worst-Case (Dense):** $O(|\Sigma|^{k+1})$. If the Trie contains every possible string of length $k$, the structure grows exponentially with the alphabet size. Total nodes:
    
    $$|\Sigma| \left( \sum_{i=0}^{k} |\Sigma|^i \right) = |\Sigma| \left( \frac{|\Sigma|^{k+1} - 1}{|\Sigma| - 1} \right) \approx |\Sigma|^{k+1}$$
    
- **Worst-Case (Sparse):** $O(nk |\Sigma|)$. If all $n$ words share no common prefixes, we have $n$ separate paths of length $k$. Each node in these paths allocates an array of size $|\Sigma|$ to store pointers for the alphabet, leading to high memory overhead.
    

[Image showing memory waste in a trie node with a large alphabet]

### Summary Comparison Table

| **Feature**             | **Multiway Trie**   | **Binary Search Tree (Strings)** |
| ----------------------- | ------------------- | -------------------------------- |
| **Search Time**         | $O(k)$              | $O(k \log n)$                    |
| **Space Complexity**    | $O(nk\|\Sigma\|)$   | $O(nk)$                          |
| **Prefix Matching**     | Excellent / Natural | Difficult                        |
| **Alphabet Dependence** | Highly Dependent    | Independent                      |

---
## Ternary Search Tree (TST)

### Summary Description

A **Ternary Search Tree** is a hybrid data structure that combines the space efficiency of a Binary Search Tree with the prefix-searching capabilities of a Trie. Each node stores a single character and has at most three children.

#### Node Structure and Logic

- **Left Child:** Points to a node whose character value is **less than** the current node.
    
- **Right Child:** Points to a node whose character value is **greater than** the current node.
    
- **Middle Child:** Points to a node representing the **next character** in the current word string.
    
- **Word Representation:** A word is formed by collecting the characters of nodes where you take the "middle" path, ending at a designated "word node."
    

### Complexity Analysis

#### Performance Metrics

- **Worst Case ($O(n)$):** Like a standard BST, if keys are inserted in alphabetical or reverse-alphabetical order, the tree can degenerate into a long, linear chain (essentially a Linked List).
    
- **Average Case ($O(\log n)$):** In a well-balanced TST, the search time is logarithmic. While the alphabet size $|\Sigma|$ heavily impacts Multiway Tries, TSTs are much more resistant to large alphabets because they use BST logic at each character level.
    
- **Best Case ($O(k)$):** If the word you are looking for (length $k$) was the very first word inserted, you will follow a direct "middle-child" path with minimal left/right branching.
    

#### Space Complexity

- **$O(nk)$:** Each character of each unique word ($n \times k$) typically requires one node.
    
- **Efficiency:** TSTs are significantly more space-efficient than Multiway Tries because they only allocate nodes for characters that actually exist in the dataset, rather than allocating a full $|\Sigma|$ array for every single node.

### Summary Comparison Table

| **Feature**              | **Multiway Trie** | **Ternary Search Tree** |
| ------------------------ | ----------------- | ----------------------- |
| **Children per Node**    | Up to $\Sigma$    | 3                       |
| **Search Time (Avg)**    | $O(k)$            | $O(\log n)$             |
| **Space Complexity**     | $O(nk\|\Sigma\|)$ | $O(nk)$                 |
| **Alphabet Sensitivity** | High              | Low                     |


### Summary of String Data Structures (Quick Reference)

| **Data Structure** | **Search (Avg)** | **Space (Worst)** | **Best Use Case**                             |
| ------------------ | ---------------- | ----------------- | --------------------------------------------- |
| **Multiway Trie**  | $O(k)$           | $O(\|\Sigma\|)$   | High-speed prefix search with small alphabets |
| **TST**            | $O(\log n)$      | $O(nk)$           | Large alphabets, limited memory               |
| **BST (Strings)**  | $O(k \log n)$    | $O(nk)$           | Simple implementations, general purpose       |

---
## Disjoint Set (Union-Find)

### Summary Description

The **Disjoint Set** ADT manages a collection of non-overlapping sets. It is defined by two primary operations: **Union**, which merges two sets into one, and **Find**, which determines which set a particular element belongs to (usually by returning a "representative" or sentinel node).

#### Up-Tree Implementation

Disjoint Sets are most efficiently implemented using **Up-Trees**, where each node points to its parent rather than its children. The root of each tree serves as the sentinel node for that set.

- **Union-by-Size (Rank):** To keep trees flat, the root of the smaller set is attached to the root of the larger set. This ensures the tree height stays logarithmic.
    
- **Path Compression:** During a `Find` operation, every node along the path to the root is reattached directly to the root. This drastically flattens the tree for future operations.
    
- **Optimal Strategy:** Because Path Compression frequently changes tree heights, making "Union-by-Height" difficult to track accurately, most implementations use **Union-by-Size** combined with **Path Compression**.

### Complexity Analysis

#### Performance Metrics

- **Find:**
    
    - **Worst Case:** $O(\log n)$ (with Union-by-Size only).
        
    - **Amortized:** $\alpha(n) \approx O(1)$. The inverse Ackermann function $\alpha(n)$ grows so slowly that it is effectively constant for all practical values of $n$.
        
    - **Best Case:** $O(1)$ (if the element is already the sentinel).
        
- **Union:**
    
    - **Worst Case:** $O(\log n)$ (dominated by the two `Find` operations required to locate the roots).
        
    - **Amortized:** $\alpha(n) \approx O(1)$.
        
    - **Best Case:** $O(1)$ (if both elements provided are already roots).
        

#### Space Complexity

- **$O(n)$:** Each of the $n$ elements is stored exactly once, with each node requiring a single parent pointer (or an integer representing size if it is a root).
    


### Summary Comparison Table (Updated)
| Data Structure | Search (Avg)  | Space (Worst)      | Best Use Case                                      |
| :------------- | :------------ | :----------------- | :------------------------------------------------- |
| Multiway Trie  | $O(k)$        | $O(\|Σ\|^{(k+1)})$ | High-speed prefix search with small alphabets      |
| TST            | $O(log n)$    | $O(nk)$            | Memory-efficient prefix search for large alphabets |
| BST (Strings)  | $O(k \log n)$ | $O(nk)$            | Simple implementations, general purpose            |
| Disjoint Set   | $~O(1)*$      | $O(n)$             | Grouping elements and cycle detection in           |

---
## Graphs and Graph Representation

### Summary Description

A **Graph** is a mathematical structure consisting of a set of vertices $V$ (nodes) and a set of edges $E$ (connections). Graphs are used to model relationships between objects, such as social networks, road maps, or task dependencies.

#### Graph Classifications

- **Directed vs. Undirected:** In a directed graph, edges have a specific direction (from $u$ to $v$). In an undirected graph, an edge between $u$ and $v$ is bidirectional.
    
- **Weighted vs. Unweighted:** Weighted graphs assign a numerical value (cost, distance, etc.) to each edge.
    
- **Sparse vs. Dense:** A sparse graph has few edges (close to $|V|$), while a dense graph has many edges (approaching the maximum possible, $|V|^2$).
    
- **Simple Graphs:** We typically disallow "parallel edges" (multiple edges between the same two nodes), ensuring $|E| \le |V|^2$.

### Adjacency Matrix

An Adjacency Matrix is a $|V| \times |V|$ 2D array where the entry at row $u$ and column $v$ indicates the presence (and weight) of an edge.

#### Performance Metrics

- **Edge Lookup:** $O(1)$. You can instantly check if an edge exists by indexing the matrix.
    
- **Iterating Outgoing Edges:** $O(|V|)$. To find neighbors of $u$, you must scan the entire $u$-th row.
    
- **Space Complexity:** $O(|V|^2)$. This is constant regardless of how many edges actually exist, making it inefficient for sparse graphs.
    

### Adjacency List

An Adjacency List is an array of $|V|$ pointers, where each pointer leads to a list (usually a Linked List or Dynamic Array) of that vertex's neighbors.

#### Performance Metrics

- **Edge Lookup:** $O(|E|)$ worst-case. In a highly skewed graph where all edges originate from one node, you might have to scan the entire list of edges.
    
- **Iterating Outgoing Edges:** $O(\text{degree of } u)$. This is the most efficient way to traverse neighbors, as you only visit existing edges.
    
- **Space Complexity:** $O(|V| + |E|)$. You only store the nodes and the edges that actually exist, making it the preferred choice for sparse graphs.

### Summary Comparison Table


| Representation   | Lookup Edge | Find Neighbors | Space Complexity   | Best Use Case |
| :--------------- | :---------- | :------------- | :----------------- | :------------ |
| Adjacency Matrix | $O(1)$      | $O(\|V\|)$     | $O(\|V\|^2)$       | Dense graphs  |
| Adjacency List   | $O(\|E\|)$  | $O(degree(u))$ | $O(\|V\| + \|E\|)$ | Sparse graphs |

### Updated Master Comparison Table
| Data Structure | Search (Avg)  | Space (Worst)           | Best Use Case                                      |
| :------------- | :------------ | :---------------------- | :------------------------------------------------- |
| Multiway Trie  | $O(k)$        | $O(\|\Sigma\|^{(k+1))}$ | High-speed prefix search with small alphabets      |
| TST            | $O(\log n)$   | $O(nk)$                 | Memory-efficient prefix search for large alphabets |
| BST (Strings)  | $O(k \log n)$ | $O(nk)$                 | Simple implementations, general purpose            |
| Disjoint Set   | $~O(1)*$      | $O(n)$                  | Grouping elements and cycle detection              |
| Adjacency List | $O(deg(u))$   | $O(\|V\|+\|E\|)$        | General graph algorithms (BFS, DFS, Dijkstra)      |
