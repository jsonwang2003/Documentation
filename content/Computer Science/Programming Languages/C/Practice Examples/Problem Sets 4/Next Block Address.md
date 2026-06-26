## Task
Implement the function `next_block_addr`
- Given a `block_header*` to the start of the heap with one allocated block, return the address of the next block
- Don't modify the struct definition at the top of the file
- Don't modify the `fake_heaps.h`, `fake_heaps.o`, or `Makefile` files
### Function Signature
```c
// Given a block_header* to the start of the heap with one allocated block, 
// return the address of the next block
//
struct block_header* next_block_addr(struct block_header* start);
```

---
## Testing
### Compiling
Use the given `Makefile` to compile your program
- To use the `Makefile` just type `make` into the terminal
### Test Cases
- To test your code, you can use the variables `heap1`, `heap2`, `heap3`, `heap4`, `heap5`, which are defined in the `fake_heaps.h` header file
	- These are pointers to the start of different heaps and have different allocated blocks in each of them
### Examples
To use the heaps in testing in `main` function
```c
int main() {

    struct block_header* myNextBlock = next_block_addr(heap1);

    printf("Start of the heap addr: %p\n", (void*)heap1);
    printf("Next block addr: %p\n", (void*)myNextBlock);

    return 0;
}
```
#### Test Case 1
**Input:** heap1  
**Output:** The difference between the start of heap memory address and the returned memory address should be 64 (decimal) or 0x40 (in hexadecimal)

#### Test Case 2
**Input:** heap3  
**Output:** The difference between the start of heap memory address and the returned memory address should be 256 (decimal) or 0x100 (in hexadecimal)

---
## Code
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include "fake_heaps.h"

struct block_header {
    size_t size_status;
};

struct block_header* next_block_addr(struct block_header* start){
    uint64_t size = start->size_status >> 2;
    void* next = ((void *)start + (size << 2));
    return (struct block_header*)next;
}
```

