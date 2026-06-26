> [!ABSTRACT]
> 
> In molecular biology, finding restriction enzyme motifs in a genome is a massive string-matching problem. While naive algorithms take $O(nmk)$ time, the **Aho-Corasick Algorithm** uses a specialized finite state machine called an **Automaton** to find all occurrences of all motifs in a single $O(n)$ linear scan.

---
## 1. The Scaling Problem

Searching for millions of short motifs ($m$) within a 3-billion-base genome ($n$) is computationally expensive.
- **Naive Search:** $O(nmk)$. Each motif is searched individually across every position in the genome.

```cpp
// Naive Search
for each word w in D: 
	for each valid start position i of Q: 
		if w == Q's substring of length |w| starting at i: 
			w was found at position i of Q
```

- **[[Multiway Trie]]:** $O(nk)$. By combining motifs into a tree, we check multiple words simultaneously, but we still have to restart the search for every new starting position in the genome.

![[Pasted image 20260202103725.png]]

---
## 2. The Aho-Corasick Automaton

The **Aho-Corasick Automaton** solves the restart problem by adding "shortcuts" to a standard **Trie**. This allows the algorithm to transition between motifs without ever re-reading a character of the genome.

### Failure Links: The Error Recovery

A **Failure Link** connects a node $u$ to another node $v$ if the path to $v$ is the longest possible suffix of the path to $u$.
- **If you fail:** When the next character in the genome doesn't match any child edge, you follow the failure link.
- **Why it works:** It preserves the progress you've already made by jumping to the start of another word that shares a suffix with what you just typed.

> [!Note] Not including the full path from the root to $u$
> This will cause the **failure link** to always point to $u$

```cpp
for each node 'curr' in a BFS starting at the root:
    if curr is the root or is a child of the root:
        create failure link from curr to the root
    else:
        p = parent of curr
        c = label of edge going into curr
        x = node pointed to by p's failure link
        repeat infinitely:
            if x has a child with edge label c:
                create failure link from curr to that child of x
                break
            else if x is the root:
                create failure link from curr to the root
                break
            else:
                x = node pointed to by x's failure link
```

![[Pasted image 20260202104005.png]]

### Dictionary Links: Finding Hidden Words

Sometimes, a word exists completely inside another word (e.g., "A" exists inside "GCA"). If you are at the end of "GCA," you might miss "A" because it ended earlier.
- **Dictionary Links** point to the nearest node that represents a complete word in your database.
- When you land on a node, you follow its dictionary links to report every word that ends at that current position in the genome.
- For each node $u$, draw a link to the first word node you would encounter if you were to repeatedly traverse **failure links**. If no such word node exists, $u$ will not have a **dictionary link**.

![[Pasted image 20260202105153.png]]

---
## 3. Algorithm Summary

### Construction (Preprocessing)

1. Build a **Multiway Trie** of all motifs in your database.
2. Use a **Breadth-First Search (BFS)** to calculate failure links.
3. Calculate dictionary links to ensure no overlapping motifs are missed.

### Scanning (The Linear Scan)

The scan is $O(n)$ because the pointer in the genome only ever moves forward. If a mismatch occurs, the "state" of the automaton shifts via failure links, but the genome index remains the same.

```C++
scan(genome):
    curr = root
    for each nucleotide c in genome:
        while curr cannot move to c:
            if curr == root: break
            curr = follow_failure_link(curr)
        
        curr = move_to_child(curr, c)
        // Report every motif found ending at this position
        report_all_matches(curr) 
```

---
## 4. Performance Comparison

|**Algorithm**|**Complexity**|**Efficiency**|
|---|---|---|
|**Naive Search**|$O(nmk)$|Extremely Slow|
|**Multiway Trie**|$O(nk)$|Much Faster|
|**Aho-Corasick**|**$O(n)$**|**Optimal**|