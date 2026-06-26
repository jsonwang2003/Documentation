> [!ABSTRACT]
> 
> A **Hash Function** ($h(k)$) is a mathematical process that transforms a key into an integer representation. While its primary goal is to facilitate $O(1)$ array indexing, it is also a fundamental tool for data integrity and security.

---
## 1. Mathematical Properties of Hash Functions

To be useful in a computer science context, a hash function must adhere to specific rules:

### Property 1: Equality (The Requirement)

If two keys $k$ and $l$ are equal ($k = l$), then their hash values **must** be equal: $h(k) = h(l)$. This ensures that if you insert an item and later search for it, the function leads you to the same location.

### Property 2: Inequality (The Goal)

If $k \neq l$, it is ideal for $h(k) \neq h(l)$.

- **Collision:** When two unequal keys produce the same hash value.
- **Perfect Hash Function:** A function that is guaranteed never to produce a collision. These are rare and usually only possible when the set of keys is known in advance.

---
## 2. Real-World Application: Data Integrity

Beyond data structures, hashing is used to verify that large files (like OS installers or games) haven't been corrupted during download.

1. **The Source:** A website provides a file and its "Check Hash" (e.g., an MD5 or SHA-1 string).
2. **The Verification:** After downloading, you run a hash function on your local file.
    - **Different Hash:** The file is definitely corrupted (Property of Equality).
    - **Same Hash:** The file is almost certainly identical to the original.

---
## 3. Evaluating Hash Function Quality

### The "O(1)" Primitive Hash

For simple types (integers, characters, booleans), hashing is a simple cast.

```C++
unsigned int hashValue(unsigned char key) {
    return (unsigned int)key; // Perfect O(1) hashing
}
```

### The "O(k)" Collection Hash

For strings or lists of length $k$, a good hash function must examine **every** element. If it only looks at the first character, strings like "Apple" and "Apply" would always collide.

**Bad String Hash (Commutative):**

Summing ASCII values is "valid" but "bad" because it is commutative. "Hello" and "eHlol" would result in the same sum.

```C++
// Bad: order doesn't matter
val += (unsigned int)(key[i]); 
```

**Good String Hash (Non-Commutative):**

A robust hash function uses math where the order of elements changes the result, significantly reducing collisions.

---
## 4. The Two-Step Indexing Process

In a Hash Table of capacity $m$, we find the storage index using two distinct steps:

1. **Hash Computation:** Use a type-specific hash function to get a large integer ($hashValue = h(key)$).
2. **Compression (The "Second Hash"):** Use the modulo operator to fit that value into the array bounds: $index = hashValue \pmod m$.

> [!WARNING]
> 
> Even with a perfect hash function, collisions can still occur during the **Compression** step if two different hash values result in the same remainder when divided by $m$.

---
## 5. Summary of "Good" Hash Design

|**Feature**|**Requirement**|**Reason**|
|---|---|---|
|**Determinism**|Absolute|Same input must always produce the same output.|
|**Coverage**|$O(k)$|Must iterate over all elements of a collection to avoid high collision rates.|
|**Math**|Non-commutative|The sequence/order of elements must influence the output.|
|**Speed**|Fast|The function should be as efficient as possible while remaining robust.|
