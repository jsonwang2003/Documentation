## Task
Implement the function `count_free_blocks`
- Given a `block_header*` to the start of the heap, return the number of free blocks
- Don't modify the struct definition at the top of the file
- Don't modify the `fake_heaps.h`, `fake_heaps.o`, or `Makefile` files
### Function Signature
```c
// Given a `block_header*` to the start of the heap, return the number of 
// free blocks
//
int32_t count_free_blocks(struct block_header* start);
```

---
## Testing
### Compiling

- Use the given `Makefile` to compile your program
    - To use the `Makefile` just type `make` into the terminal (Refer back to Week 3 Lab for more information on `Makefiles`)

### Test Cases

- To test your code, you can use the variables `heap1`, `heap2`, `heap3`, `heap4`, `heap5`, which are defined in the `fake_heaps.h` header file.
    - These are pointers to the start of different heaps and have different allocated blocks in each of them
- For example, to use `heap1` in testing in your `main` function, here's what it could look like
```c
int main() {

    int32_t numFreeBlocks = count_free_blocks(heap1);

    printf("Number of free blocks: %d\n", numFreeBlocks);

    return 0;
}
```

### Example
#### Test Case 1
**Input:** heap1  
**Output:** 0 (look into `fake_heaps.h` to see a diagram of each heap)

#### Test Case 2  
**Input:** heap3  
**Output:** 3

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

int32_t count_free_blocks(struct block_header* start){
    int32_t count = 0;
    struct block_header* current = start;
    while(current->size_status != 1){
        if(current->size_status % 2 == 0) count++;
        uint64_t size = current->size_status >> 2;
        void* next = ((void*) current + (size << 2));
        current = (struct block_header*) next;
        printf("current: %p, count: %d\n", current, count);
    }
    return count;
}
```