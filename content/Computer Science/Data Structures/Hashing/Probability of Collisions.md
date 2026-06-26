> [!ABSTRACT]
> 
> Collisions are the primary bottleneck for Hash Table performance. By using probability theory—specifically the logic behind the **Birthday Paradox**—we can determine the optimal **Hash Table Capacity ($M$)** and **Load Factor ($\alpha$)** to minimize these occurrences and maintain $O(1)$ speed.

---
## 1. The Probability of a Collision

To calculate the likelihood of a collision, it is mathematically simpler to calculate the probability that **no collision** occurs and subtract that from 1.

$$P(\text{at least 1 collision}) = 1 - P(\text{no collision})$$

As we insert $N$ keys into $M$ slots, the probability that each subsequent key avoids a collision is **conditional** on the previous keys finding empty slots:
- **1st Key:** $P = \frac{M}{M} = 1$ (100% chance of success)
- **2nd Key:** $P = \frac{M-1}{M}$ (One slot is taken)
- **3rd Key:** $P = \frac{M-2}{M}$ (Two slots are taken)

---
## 2. The Birthday Paradox

A famous illustration of this math is the **Birthday Paradox**. Even though there are $M=365$ days in a year, you don't need 365 people to likely find a shared birthday.

- With only **23 people**, there is a **50% chance** of a collision.
- With **60 people**, the chance rises to over **99%**.

**The Lesson:** Collisions happen much sooner than intuition suggests. A Hash Table that is only 16% full (60/365) is almost guaranteed to have a collision.

---
## 3. Optimal Load Factor ($\alpha$)

The **Load Factor** is defined as $\alpha = \frac{N}{M}$ (the ratio of keys to total slots). As $\alpha$ increases, the expected number of operations to find an element grows.

### The "Rule of Thumb"

- **The Threshold:** Performance remains relatively flat and fast until $\alpha \approx 0.75$.
- **The Design Choice:** To maintain $O(1)$ performance, we should aim for $M \approx 1.3N$.
- **Resizing:** If $\alpha$ exceeds 0.75 during the table's lifetime, we should resize the array (typically doubling it) and **re-hash** all existing elements.

---
## 4. Why Use Prime Numbers?

Our probability models assume that every slot is equally likely to be picked. However, if our hash function and our table capacity share common factors, we can create "dead zones" in the array that are never used, causing an unequal distribution and more collisions.

**Example of the Problem:**

- $h(k) = 3k$ (Produces only multiples of 3)
- $M = 6$ (Table size is a multiple of 3)
- Indices used: $3, 0, 3, 0...$ (Slots 1, 2, 4, and 5 stay empty forever!)

**The Solution:** Always choose a **prime number** for the table capacity $M$. Modding by a prime number helps ensure that even a patterned hash function distributes keys across all available slots.

---
## 5. Summary of Optimized Design

|**Design Parameter**|**Optimal Choice**|**Reason**|
|---|---|---|
|**Capacity ($M$)**|$\approx 1.3 \times N$|Keeps the average number of operations near constant.|
|**Load Factor ($\alpha$)**|$\le 0.75$|Prevents the drastic performance "jump" seen in crowded tables.|
|**Array Size Logic**|**Prime Numbers**|Ensures a more uniform distribution of keys across the array.|
|**Maintenance**|**Re-hash on Resize**|Updates all keys to their new valid indices when capacity changes.|
