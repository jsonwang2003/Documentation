> [!ABSTRACT]
> 
> **Double Hashing** is an advanced **Open Addressing** strategy designed to eliminate **Primary Clustering**. By using a second hash function to determine the "skip" or offset, it ensures that keys which hash to the same initial index follow different probe sequences, leading to a more uniform distribution.

---
## 1. The Core Idea: Personalized Offsets

In [[Open Addressing (Linear Probing)|Linear Probing]], the offset is always a constant 1. This leads to clumps because any key hashing to a slot near a clump gets sucked into it.

**Double Hashing** solves this by calculating a unique offset for every key. Even if two keys, $k_1$ and $k_2$, collide at the same initial index $H_1(k)$, their secondary hash $H_2(k)$ will likely be different, sending them to entirely different secondary locations.

---
## 2. The Formula

The index checked on the $i$-th attempt (starting at $i=0$) is calculated as:

$$S(k, i) = (H_1(k) + i \cdot H_2(k)) \pmod M$$

- **$H_1(k)$**: The primary hash function (determines the starting position).
- **$H_2(k)$**: The secondary hash function (determines the step size).
- **Requirement**: $H_2(k)$ must return a value greater than 0 and should ideally be relatively prime to the table size $M$ to ensure the probe sequence visits every slot.

---
## 3. Implementation

The logic remains very similar to Linear Probing, with the only change being the calculation of the `offset`.

```C++
insert_DoubleHash(k):
    index = H1(k) 
    offset = H2(k) // The personalized jump size

    Loop infinitely:
        if arr[index] == k:
            return false // No duplicates

        else if arr[index] == NULL:
            arr[index] = k
            return true // Found an empty spot!

        else:
            // Jump by the offset instead of just 1
            index = (index + offset) % M

        if index == H1(k):
            // Full circle: table is full or offset is incompatible
            enlarge_table() 
```

---
## 4. Comparison of Open Addressing Methods

| **Method**                                           | **Probe Sequence**                  | **Probability of Clumping**   |
| ---------------------------------------------------- | ----------------------------------- | ----------------------------- |
| [[Open Addressing (Linear Probing)\|Linear Probing]] | $(H_1(k) + i) \pmod M$              | **High (Primary Clustering)** |
| **Double Hashing**                                   | $(H_1(k) + i \cdot H_2(k)) \pmod M$ | **Very Low**                  |
| [[Random Hashing]]                                   | $PRNG(k)$ sequence                  | **Very Low**                  |

---

## 5. Answer: STOP and Think

**Is Double Hashing an open addressing collision resolution strategy?**

**Yes.** Like Linear Probing, Double Hashing is an **Open Addressing** strategy because the keys are stored directly within the backing array, and they are "open" to occupying any address in the table rather than being confined to a specific bucket or list (like in [[Collision Resolution/index|Separate Chaining]]).