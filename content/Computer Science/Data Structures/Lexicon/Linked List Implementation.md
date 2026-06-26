> [!ABSTRACT]
> 
> Implementing a Lexicon with a [[Computer Science/Data Structures/Introductory Data Structures/Linked List|Linked List]] is straightforward but inefficient for large datasets. Because Linked Lists lack **random access**, we are forced to perform *linear* traversals, resulting in slow lookup times that scale poorly as the dictionary grows.

---
## 1. Implementation Approaches

When using a Linked List to store words, we must choose between two organizational strategies:
### Option A: The Unsorted List

- **Insertion**: Extremely fast (**$O(1)$**). New words are simply tacked onto the `head` or `tail`.
- **Find/Remove**: Slow (**$O(n)$**). We must check every node until the word is found.
- **Trade-off**: High write speed, but data retrieval is unorganized and slow.

### Option B: The Sorted List (Alphabetical)

- **Insertion**: Slow (**$O(n)$**). We must traverse the list to find the correct alphabetical position to maintain order.
- **Find/Remove**: Still slow (**$O(n)$**). Even though the list is sorted, we cannot perform a binary search because we cannot jump to the middle of a Linked List.
- **Trade-off**: Retrieval remains slow, but the data is now organized for alphabetical iteration.

---
## 2. Performance Analysis

Regardless of the strategy chosen, the performance bottleneck remains the linear traversal.

|**Operation**|**Unsorted Complexity**|**Sorted Complexity**|
|---|---|---|
|**`find(word)`**|$O(n)$|$O(n)$|
|**`insert(word)`**|**$O(1)$**|$O(n)$|
|**`remove(word)`**|$O(n)$|$O(n)$|
|**Space**|$O(n)$|$O(n)$|

---
## 3. Evaluation for Lexicon ADT

Earlier, we established two key assumptions for our Lexicon:

1. **"Find" operations are highly frequent.**
2. **The capacity is mostly known in advance.**

**Conclusion:** The Linked List is a **poor choice** for a Lexicon. In a dictionary of 170,000 words, a "find" operation could potentially require 170,000 pointer jumps. Since lookups are our most frequent task, the $O(n)$ cost is unacceptable.