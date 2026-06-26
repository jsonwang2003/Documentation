## Task
Implement the function `split_block`
- Given a `block_header*` and a `size_t` number of bytes to allocate, return the number of free bytes left.
- Don't modify the struct definition at the top of the file
### Function Signature
```c
// Given a block_header* and a size_t number of bytes to allocate, return 
// the number of free bytes left.
//
size_t split_block(struct block_header* block, int32_t bytes_to_alloc);
```

---
## Testing
### Compiling
- Use `gcc` to compile the program and use `./` to run the program
```bash
$ gcc split_block.c -o split_block
$ ./split_block
```
- You must compile your code and get the executable file that is named exactly as above for the auto-grader to run
### Example
To test the function, create a `struct block_header` block, set the `size_status`, and plug values into the function.
#### Test Case 1
**Input:** `size_status` = 0x0000000000000080, `bytes_to_alloc` = 40  
**Output:** 80
#### Test Case 2
**Input:** `size_status` = 0x0000000000000030, `bytes_to_alloc` = 33  
**Output:** 0

---
## Code
```c
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

struct block_header {
    size_t size_status;
};

size_t split_block(struct block_header* block, size_t bytes_to_alloc){
    size_t allocd = 8 + bytes_to_alloc;
    int dif = allocd % 16;
    if(dif != 0){
            allocd += 16 - dif;
    }
    return block->size_status - allocd;
}
```80