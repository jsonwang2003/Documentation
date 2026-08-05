## Task
Implement the function `get_block_size`
- Write a function that takes a pointer to a `block_header` struct
- The `block_header` struct contains a single `size_t` field, which encodes the block size and status bits
- The block size is always a multiple of 16, so only the upper bits (all except the lowest 4 bits) represent the size. The lowest 4 bits are used for status flags and must be ignored when extracting the size
### Function Signature
```c
// Given a pointer to a block_header, return the size of the block (masking out status bits in the lower 4 bits).
size_t get_block_size(struct block_header* header);
```

---
## Example
|Encoded size_status value|Block Size Returned|
|---|---|
|0x0000000000000041|0x40 (64)|
|0x0000000000000103|0x100 (256)|
|0x0000000000000207|0x200 (512)|
|0x0000000000000801|0x800 (2048)|
|0x000000000000100f|0x1000 (4096)|

---
## Code
```c
#include <stddef.h>
#include <stdio.h>

struct block_header {
    size_t size_status;
};

size_t get_block_size(struct block_header* block) {
    int dif = block->size_status % 16;
    if (dif == 0) {return block->size_status;}
    return block->size_status - dif;
}

int main(int argc, char** argv) {
    struct block_header a = {0x444};
    printf("before: %02lx, after:  %02lx\n", a.size_status, get_block_size(&a));
    return 0;
}
```