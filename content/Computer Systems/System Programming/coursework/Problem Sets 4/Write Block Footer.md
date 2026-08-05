## Task
Implement the function `write_block_footer`
- Given a pointer to a free `block_header`, write s its footer (containing the block size) to the last 8 bytes of the block
- Remember: **ONLY FREE** blocks have footers. The footer is a `struct block_footer` containing just the size
- The footer should be placed at the very end of the block (last 8 bytes)
- Don't modify the struct definitions at the top of the file
### Function Signature
```c
// Given a pointer to a free block_header, write the footer to the last 8 bytes of the block. The footer contains only the block size (no status bits).
void write_block_footer(struct block_header* block);
```

---
## Testing
### Compiling
Use `gcc` to compile the program and use `./` to run the program
```bash
$ gcc write_block_footer.c -o write_block_footer
$ ./write_block_footer
```
### Example
To test the function, create a `struct block_header` block with a specific size, call `write_block_footer`, then check if the footer was written correctly.
```c
struct block_header block;
block.size_status = 0x0000000000000080;  // 128 bytes, free
write_block_footer(&block);
// Footer should be at address: block + 128 - 8
// Footer should contain: 0x0000000000000080
```
#### Test Case 1
**Input:** `size_status` = 0x0000000000000040  
**Expected:** Footer at offset 56 (64-8) contains 0x40
#### Test Case 2
**Input:** `size_status` = 0x0000000000000100  
**Expected:** Footer at offset 248 (256-8) contains 0x100

> [!NOTE]
> - Extract the block size using: `size = block->size_status & SIZE_MASK`
> - Calculate footer address: `(struct block_footer*)((char*)block + size - sizeof(struct block_footer))`
> - The footer's `size` field should contain the block size (without status bits)

---
## Code
```c
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define SIZE_MASK (~0xF)

struct block_header
{
    size_t size_status;
};

struct block_footer
{
    size_t size;
};

void write_block_footer(struct block_header *block)
{
    size_t size = block->size_status & SIZE_MASK;
    struct block_footer* footer_addr = (struct block_footer*)((char*) block + size - sizeof(struct block_footer));
    footer_addr->size = size;
}
```