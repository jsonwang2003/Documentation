---
title: "Cuckoo Hashing" 
description: "Collision resolution strategy where an inserting key evicts whichever key already occupies its slot, cascading displacements — trading occasional expensive rehashes for guaranteed O(1) worst-case find and delete." 
tags:
- CS/data-structures
- CS/hashing 
aliases: ["Cuckoo Hash Table"]
---

> [!abstract] Abstract 
> **Cuckoo Hashing** is an open-addressing collision resolution strategy where, instead of the _inserting_ key searching for a free slot, it evicts whichever key is already there — the displaced key then hashes to its own alternate location, potentially evicting someone else in turn. The name comes from cuckoo chicks pushing other eggs out of the nest.
> 
> - **Category:** Hash Table / Collision Resolution
> - **Stores:** keys (or key-value pairs), each with exactly two candidate slots across two hash tables
> - **Built on top of:** ordinary hash tables/arrays — this is a strategy layered on top of the standard hash-table structure
> - **Typical use cases:** anywhere a _guaranteed_ worst-case constant time for lookups and deletions matters more than a guaranteed worst-case insert time — e.g. real-time systems, hardware/router lookup tables

---

# Core Structure

Cuckoo Hashing uses **two hash functions**, $H_1(k)$ and $H_2(k)$, and (traditionally) **two hash tables**, $T_1$ and $T_2$. $H_1$ hashes keys exclusively to $T_1$; $H_2$ hashes keys exclusively to $T_2$. A key $k$ starts by hashing into $T_1$; if another key later collides with it there, $k$ gets evicted and re-hashes into $T_2$ via $H_2$. A key can also get evicted from $T_2$, in which case it hashes back into $T_1$ — potentially evicting yet another key, and so on.

> [!tip] Key Idea 
> Every key has **exactly two** possible locations in the whole structure — $H_1(k)$ in $T_1$, or $H_2(k)$ in $T_2$ — never anywhere else. That's the opposite trade-off from linear probing or double hashing, where a key could in principle end up almost anywhere: Cuckoo Hashing gives up flexibility of placement in exchange for being able to check _exactly two_ fixed spots to know for certain whether a key is present.

## Properties

- **Invariant:** at all times, every key currently in the structure sits at either $H_1(k)$ in $T_1$ or $H_2(k)$ in $T_2$ — never elsewhere. This holds by construction: `insert` only ever writes a key into one of those two computed slots, and always fully evicts (rather than merges with) whatever was previously there.
- **Shape guarantee:** exactly 2 candidate locations per key, regardless of table load.
- **Space complexity:** $O(n)$ — two backing arrays, each of capacity $M$ (or, with $d$ tables, each of capacity $M/d$).
- **What it does NOT guarantee:** worst-case constant-time **insertion** — a bad sequence of evictions can cycle, forcing a full rehash, which is $O(n)$.

## Why the Invariant Holds

Every call to `insert` writes the current key into `arr1[H1(current)]` or `arr2[H2(current)]`, saving whatever was previously in that slot as `oldValue` and continuing with `current = oldValue`. So at every step of the loop, the key being placed lands at exactly its own hash-determined slot, and whatever gets displaced is carried forward to be placed at _its_ own hash-determined slot next. No key is ever written anywhere other than its $H_1$ or $H_2$ location — the invariant isn't something that needs a separate inductive proof so much as a direct consequence of what the insert loop is literally doing at each step.

---

# Data Structure Operations

## `insert(k)`

Insert key `k`, possibly displacing an existing key, which is then re-inserted at its alternate location — cascading until some slot is empty, or a `MAX` iteration limit is hit.

- **Time complexity:** $O(1)$ average case; $O(n)$ worst case (triggers a full rehash)
- **Notes:** loops at most `MAX` times (commonly 10) before giving up and signaling that a rehash is needed.

```pseudo
	\begin{algorithm}
	\caption{Cuckoo Hash Insert}
	\begin{algorithmic}
		\Procedure{insert}{$k$}
			\State $index1 = H_1(k)$, $index2 = H_2(k)$
			\If{$arr1[index1] = k$ or $arr2[index2] = k$}
				\Return \False \Comment{duplicate}
            \EndIf
            \State $current = k$
            \While{looping fewer than $MAX$ times}
	            \State $oldValue = arr1[H_1(current)]$
	            \State $arr1[H_1(current)] = current$
	            \If{$oldValue = \text{NULL}$}
		            \Return \True
                \EndIf
                \State $current = oldValue$
                \State $oldValue = arr2[H_2(current)]$
                \State $arr2[H_2(current)] = current$
                \If{$oldValue = \text{NULL}$}
	                \Return \True
                \EndIf
                \State $current = oldValue$
            \EndWhile
            \Return \False \Comment{insertion failed; rehash needed}
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `find(k)`

Check only the two possible locations — if `k` isn't at either, it isn't in the table at all.

- **Time complexity:** $O(1)$ **worst case**
- **Notes:** this worst-case guarantee (not just average-case) is the entire point of accepting Cuckoo Hashing's more complex insert logic.

## `delete(k)`

If `k` is present, it's at $H_1(k)$ or $H_2(k)$ — remove it from whichever slot it occupies.

- **Time complexity:** $O(1)$ **worst case**
- **Notes:** same reasoning as `find` — only two slots ever need checking.

---

# Common Pitfalls

- **Choosing a bad pair of hash functions.** If $H_2$ is just a fixed shift of $H_1$ (e.g. $H_1(k) = k \bmod M$ and $H_2(k) = (k+3) \bmod M$), then two keys that collide under $H_1$ (i.e. $k_1 \equiv k_2 \pmod M$) will _also_ collide under $H_2$, since adding the same constant $3$ to both preserves the congruence. Any two keys that collide in $T_1$ are then guaranteed to collide again in $T_2$ — they'll cycle forever trying to evict each other, and the table can never actually hold both. Good $H_1$/$H_2$ pairs need enough independence that keys colliding under one are spread apart under the other.
- **No iteration cap.** Without the `MAX` limit, a bad eviction sequence can cycle indefinitely once both tables are sufficiently full — see the "Fun Fact" below for how this is typically proven.
- **Rehashing with a "shuffled" version of the same functions** instead of genuinely new ones — this can reproduce the same collision structure and immediately cycle again.

> [!note] 
> Proofs about cycles in Cuckoo Hashing are often done by converting the two hash tables' keys into a graph: keys become nodes, and each key's two possible hash locations become edges — turning "will this insertion cycle forever?" into a graph-theory question.

---

# Tradeoffs Compared to Other Data Structures

| Structure                                                    | find                         | insert                                | delete                       | Notes                                                                             |
| ------------------------------------------------------------ | ---------------------------- | ------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------- |
| **Cuckoo Hashing**                                           | $O(1)$ worst case            | $O(1)$ average, $O(n)$ worst (rehash) | $O(1)$ worst case            | Only strategy here with worst-case guarantees on find/delete                      |
| [[Open Addressing (Linear Probing)\|Linear Probing]]         | $O(1)$ average, $O(n)$ worst | $O(1)$ average, $O(n)$ worst          | $O(1)$ average, $O(n)$ worst | A key could end up scanning the whole table in the worst case                     |
| [[Closed Addressing (Separate Chaining)\|Separate Chaining]] | $O(1+\alpha)$ average        | $O(1)$                                | $O(1+\alpha)$ average        | $\alpha$ = load factor; degrades gracefully but never gives worst-case guarantees |
| [[Double Hashing]]                                           | $O(1)$ average, $O(n)$ worst | $O(1)$ average, $O(n)$ worst          | $O(1)$ average, $O(n)$ worst | Similar profile to linear probing, better probe distribution                      |

> [!note] When to reach for this structure 
> Use Cuckoo Hashing when a **guaranteed** worst-case bound on lookups and deletions is required (not just a good average case) — e.g. real-time systems or hardware routing tables — and the occasional expensive rehash on insert, plus the memory overhead of two tables, is an acceptable trade-off.

Cuckoo Hashing isn't restricted to exactly two tables — with $d$ tables, each has capacity $M/d$, where $M$ is the capacity a single-table strategy would have used.

---

# Related Notes
- [[Hash Tables]]
- [[Open Addressing (Linear Probing)|Linear Probing]]
- [[Double Hashing]]
- [[Open Addressing (Linear Probing)|Open Addressing]]