> [!ABSTRACT]
> 
> A **Bloom Filter** is a space-efficient probabilistic data structure used to test whether an element is a member of a set. Unlike a standard **Hash Table**, it can return **False Positives** but never **False Negatives**. It is the ideal solution when memory is limited and a small margin of error is acceptable.

---
## 1. The Memory Problem

Imagine you need to store **1 million** malicious URLs to protect a browser.
- **Hash Table Approach:** Storing 1 million strings (approx. 100 bytes each) at a 0.75 load factor would require over **130 MB** of RAM.
- **Bloom Filter Approach:** By storing only bits instead of actual data, you can achieve the same protection using only a few **megabytes**.

### The Trade-off: False Positives
In a security extension, the four possible outcomes of a check are:
1. **True Positive (TP):** Malicious site blocked. (Success!)
2. **True Negative (TN):** Safe site allowed. (Success!)
3. **False Negative (FN):** Malicious site allowed. (**Critical Failure!**)
4. **False Positive (FP):** Safe site blocked. (Minor Inconvenience).

A Bloom Filter is perfect here because it guarantees **zero False Negatives**. If the filter says a site is safe, it is definitely safe. If it says a site is malicious, it is _probably_ malicious.

---
## 2. How it Works: The Mechanism

A Bloom Filter consists of:
- A **Bit Array** of size $m$, initialized to all zeros (0)
	- Similar to [[Hash Tables]]
- **$k$ different Hash Functions**, each mapping an input to one of the $m$ array indices.

### Insertion Logic
To insert an element $x$:
1. Feed $x$ into all $k$ hash functions to get $k$ indices.
2. Set the bits at those $k$ indices to **1**.

### Find Logic (Membership Test)
To check if $x$ exists:
1. Compute the $k$ indices using the same hash functions.
2. **If ANY bit at those indices is 0:** The element is **definitely not** in the set.
3. **If ALL bits are 1:** The element **might** be in the set (or we encountered a False Positive due to bit overlap).

---

## 3. Mathematical Optimization

The probability of a False Positive ($\epsilon$) depends on $m$ (bits), $n$ (elements), and $k$ (hash functions):

$$\epsilon \approx (1 - e^{-kn/m})^k$$

To minimize errors when designing a filter, we use these optimal formulas:

- **Optimal number of hash functions:** $k = \frac{m}{n} \ln(2)$
- **Optimal array size:** $m = -\frac{n \ln(\epsilon)}{(\ln(2))^2}$

---

## 4. Pseudocode Implementation

### `insert(x)`

```C++
insert(x): 
    for each hash function h_i:
        index = h_i(x) % m
        bit_array[index] = true
```

### `find(x)`

```C++
find(x):  
    for each hash function h_i:
        index = h_i(x) % m
        if bit_array[index] == false:
            return false // DEFINITELY NOT PRESENT
    return true // POSSIBLY PRESENT
```

---
## 5. Summary Comparison

|**Feature**|**[[Hash Tables]]**|**Bloom Filter**|
|---|---|---|
|**Storage**|Stores actual keys|Stores only bits (0/1)|
|**Memory Usage**|High ($O(n \times \text{key\_size})$)|Very Low ($O(m)$)|
|**Search Time**|$O(1)$ Average|$O(k)$ (Constant $k$ functions)|
|**False Positives**|No|**Yes**|
|**False Negatives**|No|**No**|
