> [!ABSTRACT]
> 
> Randomness is a core pillar of modern computer science, essential for Randomized Algorithms (Las Vegas and Monte Carlo) and Randomized Data Structures. At its simplest level, all computational randomness is reduced to random number generation.

---
## 1. Classification of Randomized Algorithms

Algorithms that use randomness are generally divided into two categories based on their guarantees:

|**Type**|**Accuracy Guarantee**|**Performance Guarantee**|
|---|---|---|
|**Las Vegas**|**Guaranteed** to be correct.|Time/memory usage varies based on random input.|
|**Monte Carlo**|Chance of being **incorrect**.|Running time is usually fixed or strictly bounded.|

---
## 2. True vs. Pseudo-Randomness

Generating "pure" randomness is a significant challenge for deterministic machines like computers.

### True Random Number Generation (TRNG)
- **Mechanism**: Measures physical phenomena (atmospheric noise, thermal noise, quantum decay) to harvest **natural entropy**.
- **Pros**: Truly non-deterministic and unpredictable.
- **Cons**: **Slow**. The computer must often "block" (wait) until enough physical entropy is gathered to meet demand.
### Pseudo-Random Number Generation (PRNG)
- **Mechanism**: Uses mathematical algorithms to produce a sequence of numbers based on an initial value called a **Seed**.
- **Pros**: **Extremely fast** and computationally efficient.
- **Cons**: **Deterministic**. If you use the same seed, you will get the exact same sequence of "random" numbers every time.

---
## 3. The Hybrid Approach
To achieve a balance between speed and quality, modern systems typically use a **hybrid method**:
1. Harvest a single **True Random Number** (e.g., from system time or thermal noise).
2. Use that true random number as the **Seed** for a **Pseudo-Random Number Generator**.
3. Generate a long sequence of numbers quickly.

This ensures that the sequence is different every time the program runs without the performance penalty of generating every single digit from physical entropy.

---
## 4. Implementation in C++

In C++, basic randomness is handled by the `<cstdlib>` and `<ctime>` libraries.

### Basic Syntax

To prevent the same sequence from appearing every run, we seed the generator once at the start of the program using `srand()`.

```cpp
#include <cstdlib>
#include <ctime>

// 1. Seed the generator using the current system time
srand(time(NULL));

// 2. Generate a random number between 0 and RAND_MAX
int rawRandom = rand(); 
```

### Controlling the Range

We can map the raw random output to a specific range using the **modulo (%)** operator.

|**Desired Range**|**Code Logic**|**Explanation**|
|---|---|---|
|**0 to $N-1$**|`rand() % N`|Remainder is always less than $N$.|
|**1 to $N$**|`(rand() % N) + 1`|Shifts the 0-indexed range up by 1.|
|**Random Index**|`rand() % myVec.size()`|Safely picks a valid index for a vector/array.|

---
## 5. Summary
- **TRNGs** are slow but unpredictable; **PRNGs** are fast but deterministic.
- High-quality randomness is achieved by **seeding a PRNG with a TRNG**.
- Any complex probability distribution (Gaussian, Poisson, etc.) can eventually be derived from a simple **Uniform Distribution** of random integers.