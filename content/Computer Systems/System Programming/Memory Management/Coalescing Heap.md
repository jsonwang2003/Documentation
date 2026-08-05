> [!ABSTRACT]
> 
> This section explores how a memory allocator reclaims space on the heap. We focus on the Fragmentation Problem—where free memory exists but is unusable because it is split into small pieces—and the solution of Coalescing adjacent free blocks.

### 1. The Fragmentation Problem
When memory is allocated and freed in a non-linear order, "holes" of free memory appear between allocated blocks. Even if the total amount of free memory is large enough for a new request, a standard allocator may fail if those holes are not contiguous (touching).

**The Scenario:**

```c
a = malloc(16);
b = malloc(16);
c = malloc(16);
d = malloc(16);
free(b); // 16 bytes freed at 'b'
free(c); // 16 bytes freed at 'c'
e = malloc(32); 
```

In our current implementation from [[Implementation of Malloc]], we cannot use the freed space to store `e` because the individual blocks `b` and `c` do not recognize the combined space as a single usable block. `b` is too small, and `c` is too small, so `e` is forced to the end of the heap behind `d`.

---
### 2. Solution: Coalescing Adjacent Free Blocks
**Coalescing** is the process of merging adjacent free blocks into a single larger block.
#### Implementation: `find_block` with Coalescing
To implement this, we redefine `find_block` to merge blocks on-the-fly when searching for space.

```c
uint64_t find_block(uint64_t* start, size_t size){
    int i = 0;
    while(i < HEAP_WORDS){
        uint64_t header = start[i];
        int nexti = i + (header / 8);
        uint64_t nexth = start[nexti];
        
        // Logic: Is the current block AND the next block free?
        if(header % 2 == 0 && nexth % 2 == 0){
            // Merge them by adding their sizes
            start[i] = header + nexth;
            continue; // Re-check this index to see if more blocks can merge
        }
        
        // Return the address if the block is free and large enough
        if(header % 2 == 0 && header >= size){
            return &start[i];
        }
        i += header / 8; // Jump to the next block
    }
    return NULL;
}
```

**Key Logic Checks:**
- **Is the block free?** `header % 2 == 0`.
- **Is the size sufficient?** `header >= size`.
- **Is the next block free?** Locate `nexti` by adding the current size (converted to words) to the current index, then check `nexth % 2 == 0`.

---
### 3. Malloc Design Trade-offs
Allocators must balance how efficiently they use memory against how fast they respond to requests.

|**Strategy**|**Logic**|**Pros**|**Cons**|
|---|---|---|---|
|**First Fit**|Use the _earliest_ big enough block.|**Speed**: Stops searching quickly.|Can leave many small "splinters" at the start of the heap.|
|**Best Fit**|Use the _smallest_ big enough block.|**Utilization**: Maximizes the density of the heap.|**Slow**: Must inspect every block in the heap.|

**Metrics for Success:**
- **Utilization**: What percentage of the heap is used versus wasted as extra free space between the first and last allocated blocks?
- **Speed**: How many blocks must be inspected per allocation call?

---

### 4. Advanced Memory Functions: `realloc` and `calloc`

#### `realloc(ptr, new_size)`
Resizes the block at `ptr`.
- It attempts to expand the block in its current location if possible.
- If the current location is blocked, it returns a newly-allocated block elsewhere and copies the data.

![[Pasted image 20251117102406.png]]

#### `calloc(count, size)`
Performs the same task as `malloc` but with one key difference: it sets all the newly allocated data to **zero** (`0`).

---
### Module Navigation
- **[[Implementation of Malloc]]**: The internal header and word structure this coalescing logic relies on.
- **[[Virtual Memory]]**: How the OS maps these heap addresses to physical hardware.