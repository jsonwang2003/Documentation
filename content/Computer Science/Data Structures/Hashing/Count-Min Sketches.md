> [!ABSTRACT]
> 
> A **Count-Min Sketch** is a space-efficient, probabilistic data structure that functions like a frequency table. While a [[Hash Tables|Hash Map]] stores every key-value pair, the Count-Min Sketch uses a fixed-size 2D array to provide an **over-estimate** of an element's frequency. It is the "big brother" of the [[Bloom Filters|Bloom Filter]], trading exact counts for massive memory savings in high-volume data streams.

---
## 1. The Memory Wall: Netflix Scale

Imagine tracking the view counts of every episode on Netflix over 48 hours.
- **The Hash Map Problem:** Storing millions of unique episode IDs as keys and their 64-bit integer counts as values would consume gigabytes of RAM. As the lead engineer, your system would crash under the weight of this metadata.
- **The Solution:** Instead of storing the keys themselves, we use a **Count-Min Sketch**. It allows us to track frequencies using a constant amount of memory, regardless of how many unique episodes exist.

---
## 2. Structure and Mechanism

A Count-Min Sketch consists of a 2D array (matrix) with:
- **$k$ rows:** Each row corresponds to a unique hash function.
- **$m$ columns:** The range of each hash function.

![[Pasted image 20260202102456.png]]
### Incrementing a Count

To record an event (e.g., a user watches an episode of _Friends_):

1. Pass the episode ID through each of the $k$ hash functions.
2. Each function $h_i$ gives you a column index for its specific row.
3. **Increment** the counter in each of those $k$ cells.
### Estimating a Count (The "Find" Operation)

Because different keys might hash to the same cells (**collisions**), the values in the cells can be higher than the actual count of a specific key.

1. Retrieve the values from the $k$ hashed positions.
2. The **minimum** of these $k$ values is your estimate.

> [!IMPORTANT] **Why the minimum?** 
> Since collisions only ever **increase** the values in the cells, the smallest value among all $k$ rows is guaranteed to be the "cleanest" estimate. While this minimum can still be an over-estimate, the true count $c_x$ will **never** be greater than this value.

---
## 3. Mathematical Design

To minimize the error in our estimates, we design the dimensions of the sketch based on our tolerance for error ($\epsilon$) and our desired confidence level ($1-\delta$):
- **Width (Columns $m$):** $m = \lceil \frac{e}{\epsilon} \rceil$. More columns reduce the chance of collisions in any single row.
- **Depth (Rows $k$):** $k = \lceil \ln(\frac{1}{\delta}) \rceil$. More rows (hash functions) decrease the probability that every row will have a significant collision for a specific key.

---
## 4. Pseudocode

### `increment(x)`

```C++
increment(x):
    for i from 0 to k-1:
        column = hash_functions[i](x) % m
        matrix[i][column] += 1
```

### `estimate(x)`

```C++
estimate(x):
    min_val = infinity
    for i from 0 to k-1:
        column = hash_functions[i](x) % m
        current_val = matrix[i][column]
        if current_val < min_val:
            min_val = current_val
    return min_val
```

---
## 5. Summary Comparison

|**Feature**|**[[Hash Tables\|Hash Map]]**|**[[Count-Min Sketches\|Count-Min Sketch]]**|
|---|---|---|
|**Accuracy**|100% Precise|Probabilistic (Over-estimates)|
|**Memory**|$O(n)$ — Grows with unique keys|$O(m \times k)$ — Fixed size|
|**Keys Stored**|Yes|No|
|**Best Use Case**|Small/Medium datasets|Heavy-hitter detection in massive streams|