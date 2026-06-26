> [!ABSTRACT]
> 
> A **[[Computer Science/Data Structures/Hashing/index|Hash Table]]** implementation offers the fastest average-case performance for the Lexicon ADT. By transforming a word into a numerical index via a **Hash Function**, we can achieve **$O(1)$** average-case lookup, insertion, and removal.

---
## 1. Mechanics: From Words to Indices
To store a word in a Hash Table, the system performs two steps:
1. **Hashing:** A function processes the string (word) of length $k$ to produce a "Hash Value." This takes **$O(k)$** time.
2. **Indexing:** The hash value is mapped to an index in the backing array (usually via the modulo operator: `index = hash % array_size`).

---
## 2. Performance Analysis

While we often call Hash Table operations $O(1)$, in the context of a Lexicon, we must acknowledge the time spent processing the string itself ($k$).

|**Operation**|**Complexity (Average)**|**Logic**|
|---|---|---|
|**`find(word)`**|**$O(k)$**|Time to hash a string of length $k$ + $O(1)$ jump.|
|**`insert(word)`**|**$O(k)$**|Time to hash + $O(1)$ placement.|
|**`remove(word)`**|**$O(k)$**|Time to hash + $O(1)$ deletion.|
|**Space**|**$O(n)$**|Requires extra "padding" to keep the **Load Factor** low (typically ~70%).|

---
## 3. Lexicon vs. Dictionary

By upgrading from a **Hash Table** to a **Hash Map**, we transition from a simple word list to a full **Dictionary**:
- **Key:** The word (used for hashing).
- **Value:** The definition, etymology, or metadata.
- **Benefit:** We gain full dictionary utility with no significant loss in performance.

---
## 4. Evaluation for Lexicon ADT

The Hash Table is a powerful contender for digital lexicons, but it comes with specific trade-offs:
- **Speed:** It is faster than Binary Search on average. A word with 7 letters takes roughly 7 "steps" to hash, regardless of whether the dictionary has 100 or 1,000,000 words.
- **Ordering:** Unlike Sorted Arrays or BSTs, Hash Tables are **unordered**. You cannot easily print the lexicon in alphabetical order or find the "next" word in the sequence.
- **Memory Waste:** To prevent collisions (where two words map to the same index) and maintain $O(1)$ speed, we must leave roughly 30% of the array empty.
- **Worst-Case:** In the absolute worst-case (where many words collide), performance can degrade to $O(n)$.

---
## 5. Comparison: Sorted Array vs. Hash Table

|**Feature**|**Sorted Array**|**Hash Table (Average)**|
|---|---|---|
|**Search Speed**|$O(\log n)$|**$O(k)$**|
|**Alphabetical Order**|**Yes**|No|
|**Space Efficiency**|High (Compact)|Lower (Needs ~30% empty space)|
|**Worst-Case Search**|$O(\log n)$|$O(n)$ (though rare)|