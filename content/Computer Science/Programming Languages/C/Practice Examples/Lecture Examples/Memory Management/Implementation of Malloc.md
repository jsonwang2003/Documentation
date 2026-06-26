> [!ABSTRACT] 
> Implementing `malloc` involves requesting a large segment of memory from the Operating System and manually partitioning it into "blocks." Each block uses a **header** to store metadata about its size and whether it is currently in use (Busy vs. Free).

### 1. Initializing the Heap
To start, the allocator must request a large chunk of memory from the OS. While the older `sbrk` is deprecated, modern implementations use `mmap`.

```c
void init_heap(){
    // Request a chunk of anonymous, private, read/write memory from the OS
    HEAP_START = mmap(NULL, HEAP_SIZE_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    // Initialize the very first header to represent the total heap size
    HEAP_START[0] = HEAP_SIZE_BYTES; 
}
```

- **The Header**: In this implementation, the header stores the total size of the block in bytes.
- **LSB Trick**: Since block sizes are aligned to 8 bytes, the least significant bit (LSB) is always 0. We use this bit as a **flag**: `0` for Free, `1` for Busy.

---
### 2. Block Management Logic
The allocator performs three main tasks: finding space, splitting blocks to minimize waste, and freeing memory.
#### Finding a Block (`find_block`)
The allocator traverses the heap word by word, checking headers to find a free block large enough for the request.

```c
uint64_t* find_block(uint64_t* start, size_t size){
    int i = 0;
    while(i < HEAP_WORDS){
        uint64_t header = start[i];
        if (header >= size && header % 2 == 0){ // Size is enough and LSB is 0 (Free)
            return &start[i];
        }
        i += header / 8; // Pointer arithmetic: jump to the next header
    }
    return NULL;
}
```

#### Splitting a Block (`split_block`)
If a found block is larger than needed, it is split into a **Busy** block for the user and a new **Free** block for the remainder.
- **Busy Header**: `size + 1` (the +1 sets the Busy flag).
- **Remaining Free Header**: `header - size`.
#### Freeing Memory (`free`)
To free memory, the allocator moves back one word from the user's pointer to reach the header and flips the Busy bit back to 0 using a bitwise `AND` with a mask (`~1`).

```c
void free(void* ptr){
    uint64_t* p = ptr;
    p[-1] = p[-1] & (~1); // Access header via negative indexing and mark as Free
}
```

---
### 3. Byte Alignment and Word Math
C requires that memory returned by `malloc` be aligned (usually to 8 or 16 bytes) for CPU efficiency.
- **Block Size Calculation**: The function `block_size_of` ensures the request is rounded up to the nearest 8 bytes, plus an additional 8 bytes for the header.
- **Pointer Arithmetic**: When navigating `uint64_t*`, adding 1 to the pointer moves it 8 bytes forward in memory.

![[6fa4d2a9-94df-4e2f-962f-b84aba50febb.png]]

---
### 4. Implementation Trace
When we allocate and free in sequence, we can see the heap structure evolve:
1. **Allocation**: Blocks are marked **Busy** (e.g., `25` represents 24 bytes + Busy flag).
2. **Freeing**: A Busy block (49) becomes Free (48) after calling `free()`.
3. **Reuse**: A subsequent `malloc` can now take over that 48-byte free spot, potentially splitting it further.