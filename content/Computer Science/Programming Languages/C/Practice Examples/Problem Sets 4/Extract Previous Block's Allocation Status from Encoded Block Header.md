## Task
Implement the function `get_prev_status`
- Write a function that takes a pointer to a `block_header` struct
- The `block_header` struct contains a single `size_t` field called `size_status`, which encodes the block size and status bits
- The second lowest bit (bit 1) of the `size_status` field indicates whether the previous block is allocated (1) or free (0)
- Your function should return `1` if the previous block is allocated or `0` if it is free
### Function Signature
```c
// Given a pointer to a block_header, return 1 if the previous block is allocated, 0 if it is free.
int get_prev_status(struct block_header* header);
```

---
## Example
| Encoded size_status value | Prev Allocated? (Output) |
| ------------------------- | ------------------------ |
| 0x0000000000000040        | 0                        |
| 0x0000000000000042        | 1                        |
| 0x0000000000000041        | 0                        |
| 0x0000000000000043        | 1                        |
| 0x0000000000000100        | 0                        |
| 0x0000000000000102        | 1                        |
| 0x0000000000000101        | 0                        |
| 0x0000000000000103        | 1                        |
| 0x0000000000000200        | 0                        |
| 0x0000000000000202        | 1                        |
| 0x0000000000000201        | 0                        |
| 0x0000000000000203        | 1                        |
| 0x0000000000001000        | 0                        |
| 0x0000000000001002        | 1                        |
| 0x0000000000001001        | 0                        |
| 0x0000000000001003        | 1                        |

---
## Code
```c
#include <stddef.h>

struct block_header {
    size_t size_status;
};

size_t get_prev_status(struct block_header* block) {
    size_t dif = block->size_status >> 1;
    if (dif % 2 == 1) return 1;
    return 0;
}
```