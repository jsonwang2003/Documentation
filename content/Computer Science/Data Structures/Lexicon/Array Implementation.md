> [!ABSTRACT]
> 
> An **[[Array Lists|Array]]** implementation of a Lexicon relies on **Random Access** to enable **[[Computer Science/Discrete Structures/Discrete Algorithms/Searching/Binary Search]]**. While it incurs a high cost for modifications ($O(n)$ to shift elements), it is a superior choice for a Lexicon where word lookups are frequent and the dictionary remains relatively static.

---
## 1. Why Sorting Matters

In a Lexicon, an unsorted array is no better than a [[Linked List Implementation]] (requiring a $O(n)$ Linear Search). However, a **Sorted Array** changes the math:
- **Random Access:** Because array slots are contiguous in memory, we can calculate the address of any index in $O(1)$ time.
- **Binary Search:** Using random access, we can check the middle element, discard half the list, and repeat. This reduces the search space from $n$ to $1$ in just $\log_2 n$ steps.

---
## 2. Performance Analysis

Because we keep the array sorted and compact (no gaps), our complexity reflects the cost of maintaining that order.

| **Operation**      | **Complexity**  | **Logic**                                                          |
| ------------------ | --------------- | ------------------------------------------------------------------ |
| **`find(word)`**   | **$O(\log n)$** | Enabled by **Binary Search**.                                      |
| **`insert(word)`** | $O(n)$          | Must shift existing elements to maintain alphabetical order.       |
| **`remove(word)`** | $O(n)$          | Must shift elements to fill the gap left by the deleted word.      |
| **Space**          | $O(n)$          | $n$ slots for words, plus potential overhead for dynamic resizing. |

---
## 3. Evaluation for Lexicon ADT

The Array implementation aligns well with our specific Lexicon assumptions:
- **Fast Lookups:** $O(\log n)$ is a massive improvement over $O(n)$. For a dictionary of 170,000 words, Binary Search takes about **18 comparisons**, whereas a Linked List could take **170,000**.
- **Infrequent Updates:** While $O(n)$ insertion is slow, it is acceptable in this context because we rarely add or remove words from a language.
- **Memory Efficiency:** Arrays are very space-efficient, though **Dynamic Arrays** may temporarily double their allocated space ($2n$) to accommodate growth.

---
## 4. Comparison: Linked List vs. Array

|**Feature**|**Linked List**|**Sorted Array**|
|---|---|---|
|**Search Speed**|$O(n)$|**$O(\log n)$**|
|**Random Access**|No|**Yes**|
|**Insertion**|$O(1)$ (if unsorted)|$O(n)$|
|**Memory Overhead**|Pointers for every node|Contiguous block|
