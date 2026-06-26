---
title: Lexicon
---
> [!ABSTRACT]
> 
> A Lexicon is a computational representation of a dictionary or word list. In computer science, we define it as an ADT primarily focused on efficient word management and retrieval.

---
## 1. Defining the Interface

The Lexicon ADT is defined by three fundamental operations. In most practical applications, these operations are performed on strings (words).

|**Operation**|**Description**|
|---|---|
|**`find(word)`**|Searches the lexicon to determine if a specific word exists.|
|**`insert(word)`**|Adds a new word to the lexicon.|
|**`remove(word)`**|Deletes an existing word from the lexicon.|

---
## 2. Operational Assumptions

Unlike general-purpose dynamic data structures, Lexicons often operate under specific constraints based on how language is used:
- **Read-Heavy Workload:** "Find" operations are significantly more frequent than "insert" or "remove." This is because languages are relatively stable; we look up existing words millions of times more often than we invent new ones or delete archaic ones.
- **Known Capacity:** We typically know the approximate size of the lexicon (the number of words) before we build the structure. For example, the Oxford English Dictionary contains roughly 170,000 words in current use.
- **Static Nature:** Because the underlying data (the English language) doesn't change every second, we can sometimes trade off slow insertion times for extremely fast lookup times.

---
## 3. Potential Implementation Strategies

Given the assumptions that "find" operations are frequent and the size of the lexicon is often known in advance, we must choose a **Data Structure** that optimizes for retrieval. Below are the candidates we will explore in this chapter:
### [[Linked List Implementation|Linked List]]
- **Lookup:** $O(n)$ in the worst case.
- **Suitability:** Generally inefficient for large lexicons due to linear search times.
### Arrays
- **Lookup:** $O(\log n)$ using Binary Search if kept sorted.
- **Suitability:** High performance for searching, especially since the lexicon size is known, but "insert" and "remove" operations are $O(n)$ due to element shifting.
### Binary Search Trees (BSTs)
- **Lookup:** $O(\log n)$ on average; self-balancing versions (AVL or Red-Black) guarantee $O(\log n)$ in the worst case.
- **Suitability:** A balanced choice for dynamic data, though it requires extra memory for pointers.
### Hash Tables and Hash Maps
- **Lookup:** $O(1)$ average-case time complexity.
- **Suitability:** Extremely fast for exact word matches.
### Multiway Tries
- **Lookup:** $O(L)$ where $L$ is the length of the word being searched.
- **Suitability:** Highly optimized for string operations and prefix matching (like autocomplete), though they can be memory-intensive.
### Ternary Search Trees
- **Lookup:** Efficiently balances the space-saving nature of BSTs with the prefix-searching power of Tries.
- **Suitability:** Ideal for digital lexicons where memory efficiency and fast string lookups are both priorities.