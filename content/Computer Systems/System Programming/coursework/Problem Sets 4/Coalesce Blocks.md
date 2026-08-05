## Task
Implement the function `coalesce_blocks`
- Given two adjacent `block_header*`, returning the number of free bytes formed by coalescing the two blocks
- Don't modify the struct definition at the top of the file
### Function Signature
```c
// Given two adjacent block_header*, returning the number of free bytes formed by coalescing the two blocks.
size_t coalesce_blocks(struct block_header* block1, struct block_header* block2);
```

---
## Testing
### Compiling
- Use `gcc` to compile the program and use `./` to run the program
```bash
$ gcc coalesce_blocks.c -o coalesce_blocks
$ ./coalesce_blocks
```
- You must compile your code and get the executable file that is named exactly as above for the auto-grader to run
### Example
To test the function, create a `struct block_header` block, set the `size_status`, and plug values into the function.
#### Test Case 1
**Input:** block1 = 0x0000000000000081, block2 = 0x0000000000000100  
**Output:** 384
#### Test Case 2
**Input:** block1 = 0x0000000000000200, block2 = 0x0000000000000102  
**Output:** 768

---
## Code
```c
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

struct block_header {
    size_t size_status;
};

size_t coalesce_blocks(struct block_header* block1, struct block_header* block2){
    return (block1->size_status & ~3) + (block2->size_status & ~3);
}
```