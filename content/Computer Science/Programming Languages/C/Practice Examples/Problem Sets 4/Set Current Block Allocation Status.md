## Task
Implement the function `set_curr_status`
- Given a pointer to a `block_header` and a status value (`0` or `1`), update the block's allocation status bit
- The least significant bit (bit 0) indicates status
	- 1 → allocated
	- 0 → free
- Preserve all other bits in `size_status` (size and previous block status)
- Don't modify the struct definition at the top of the file
### Function Signature
```c
// Given a pointer to a block_header and status (0 or 1), set the current block's allocation status bit while preserving all other bits.
void set_curr_status(struct block_header* block, int status);
```

---
## Testing
### Compiling
Use `gcc` to compile the program and use `./` to run the program
```bash
$ gcc set_curr_status.c -o set_curr_status
$ ./set_curr_status
```
### Example
#### Test Case 1
**Input:** `size_status` = 0x0000000000000040, status = 1  
**Expected:** `size_status` becomes 0x0000000000000041
#### Test Case 2
**Input:** `size_status` = 0x0000000000000043, status = 0  
**Expected:** `size_status` becomes 0x0000000000000042

> [!NOTE]
> - To set bit 0 to 1: `block->size_status |= 0x1`
> - To set bit 0 to 0: `block->size_status &= ~0x1`
> - Use conditional logic based on the `status` parameter

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

void set_curr_status(struct block_header *block, int status)
{
    if (status == 1){
            block->size_status |= 0x1;
    } else {
            block->size_status &= ~0x1;
    }
}
```