> [!ABSTRACT]
> 
> **Linear Probing** is a collision resolution strategy within the **Open Addressing** (or **Closed Hashing**) family. When a key's natural hash index is occupied, the algorithm "probes" the very next sequential slot in the array. This continues until an empty slot is found or the table is determined to be full.

---
## 1. The Core Concept: Sequential Search

In Linear Probing, the algorithm follows a simple deterministic path: if $H(k)$ is full, try $H(k)+1$, then $H(k)+2$, and so on.

![[Pasted image 20260206100526.png]]
### Terminology
- **Open Addressing:** The "address" of a key is not fixed; it is open to moving to a different index than its hash value.
- **Closed Hashing:** The key must stay "closed" within the boundaries of the backing array.

---
## 2. Implementation: Insert and Find
### Insertion

To insert a key $k$:
1. Calculate `index = H(k)`.
2. If `arr[index]` is empty, place $k$ there.
3. If `arr[index]` is occupied, increment the index: `index = (index + 1) % M`.
4. Repeat until an empty slot is found. If you return to $H(k)$, the table is full and must be resized.

```cpp
insert_LinearProbe(k): // Insert k into the Hash Table using Linear Probing
    index = H(k)

    Loop infinitely:
        if arr[index] == k:         // check for duplicate insertions
            return false

        else if arr[index] == NULL: // insert if slot is empty
            arr[index] = k
            return true

        else:                       // there is a collision, so recalculate index
            index = (index + 1) % M

        if index == H(k):           // we went full circle (no empty slots)
            enlarge arr and rehash all existing elements
            index = H(k)            // H(k) will index differently now that arr is a different size
```

The core of the **Linear Probing** algorithm is defined by this equation:

```cpp
index = (index + 1) % M
```
### Finding a Key

The `find` algorithm must follow the same probe sequence. It starts at $H(k)$ and scans linearly until:
1. It finds the key (**Success**).
2. It hits a `NULL` slot (**Failure**—if the key were there, it would have been placed here or before this point).

---
## 3. The Deletion Problem: Lazy Deletion

You cannot simply set a slot to `NULL` when deleting a key. Doing so would break the "chain" of probes for other keys that collided at that spot.

**Example:** If key $A$ and key $B$ both hash to index 5, and $B$ is placed at index 6 via probing, deleting $A$ by setting index 5 to `NULL` would make $B$ impossible to find (the search would stop at the new `NULL` at index 5).

### Solution: Tombstones

Instead of **deleting the element at index**, we set that index to be a "tombstone" index

We use a special "Deleted" flag (or **Tombstone**).
- **Find** treats a tombstone as an occupied slot and keeps searching.
- **Insert** treats a tombstone as an empty slot and can replace it with a new key.

---
## 4. Disadvantages: Primary Clustering

The most significant weakness of Linear Probing is **Primary Clustering**.

As slots fill up, "clumps" of keys form. These clumps are statistically more likely to grow because any key that hashes to _anywhere_ inside the clump will be forced to the end of it, making the clump even longer. This turns $O(1)$ operations into $O(N)$ linear scans.

---
## 5. Better Alternatives

To avoid clumping, we use strategies that provide a different "skip" for each key:
- **[[Double Hashing]]:** Uses a second hash function to determine the step size ($H_2(k)$).
- **[[Random Hashing]]:** Uses a pseudorandom sequence seeded by the key to determine probe positions.

---

## 6. Performance Summary

|**Feature**|**[[Open Addressing (Linear Probing)\|Linear Probing]]**|
|---|---|
|**Average Time**|$O(1)$|
|**Worst-Case Time**|$O(N)$|
|**Cache Locality**|**Excellent** (Sequental memory access)|
|**Main Weakness**|Primary Clustering|
|**Deletion Method**|Lazy Deletion (Tombstones)|