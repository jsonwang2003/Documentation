## Task
Implement the function `get_prev_block_size`
- Given a pointer to a `block_header`, read the footer of the **PREVIOUS** block to get its size
- This only works if the previous block is **FREE** (has a footer)
- The footer is located at the 8 bytes immediately before the current block's header
- Don't modify the struct definitions at the top of the file
### Function Signature
```c
// Given a pointer to a block_header, read the previous block's footer
// to get its size. Assumes the previous block is free and has a footer.
//
size_t get_prev_block_size(struct block_header* block);
```

---
## Testing
### Compiling
- Use `gcc` to compile the program and use `./` to run the program
```bash
$ gcc get_prev_block_size.c -o get_prev_block_size
$ ./get_prev_block_size
```

---
## Code
```c
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

struct block_header
{
    size_t size_status;
};

struct block_footer
{
    size_t size;
};

size_t get_prev_block_size(struct block_header *block)
{
    struct block_footer* footer = (struct block_footer*) block;
    footer -= 1;
    return footer->size;
}
```