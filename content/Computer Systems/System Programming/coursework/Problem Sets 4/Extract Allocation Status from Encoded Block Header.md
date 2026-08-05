## Task 
Implement the function `get_curr_status`
- Write a function that takes a pointer to a `block_header` struct
- The `block_header` struct contains a single `size_t` field, which encodes the block size and status bits
- The lowest bit (bit 0) of the `size_status` field indicates whether the current lock is allocated (1) or free (0)
- Your function should return `1` if the block is allocated, `0` if it is free
### Function Signature
```c
// Given a pointer to a block_header, return 1 if the block is allocated, 0 if it is free.
int get_curr_status(struct block_header* header);
```

---
## Example
|Encoded size_status value|Allocated? (Output)|
|---|---|
|0x0000000000000040|0|
|0x0000000000000041|1|
|0x0000000000000100|0|
|0x0000000000000101|1|
|0x0000000000000200|0|
|0x0000000000000201|1|
|0x0000000000001000|0|
|0x0000000000001001|1|

---
## Code
```c
#include <stddef.h>
#include <stdio.h>

struct block_header {
    size_t size_status;
};

size_t get_curr_status(struct block_header* block) {
    if(block->size_status % 2 == 1) {return 1;}
    return 0;
}

int main(int argc, char** argv) {
    struct block_header a = {0x440};
    struct block_header b = {0x443};
    printf("a->%02lx %ld\n", a.size_status, get_curr_status(&a));
    printf("b->%02lx %ld\n", b.size_status, get_curr_status(&b));
    return 0;
}
```