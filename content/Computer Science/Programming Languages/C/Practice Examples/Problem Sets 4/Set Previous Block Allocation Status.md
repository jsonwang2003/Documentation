## Task
Implement the function `set_prev_status`
- Given a pointer to a `block_header` and a status value (0 or 1), and update the "previous block busy" bit
- The second least significant bit (bit 1) indicates previous block status
	- 1 → allocated
	- 0 → free
- Preserve all other bits in `size_status` (size and current block status)
- Don't modify the struct definition at the top of the file
### Function Signature
```c
// Given a pointer to a block_header and status (0 or 1), set the previous block's allocation status bit while preserving all other bits.
void set_prev_status(struct block_header* block, int status);
```

---
## Testing
### Compiling
Use `gcc` to compile the program and use `./` to run the program
```bash
$ gcc set_prev_status.c -o set_prev_status
$ ./set_prev_status
```
### Example
#### Test Case 1
**Input:** `size_status` = 0x0000000000000041, status = 1  
**Expected:** `size_status` becomes 0x0000000000000043
#### Test Case 2
**Input:** `size_status` = 0x0000000000000043, status = 0  
**Expected:** `size_status` becomes 0x0000000000000041

> [!NOTE]
> - To set bit 1 to 1: `block->size_status |= 0x2`
> - To set bit 1 to 0: `block->size_status &= ~0x2`
> - Use conditional logic based on the `status` parameter

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

void set_prev_status(struct block_header *block, int status)
{
    if(status == 1){
            block->size_status |= 0x2;
    } else {
            block->size_status &= ~0x2;
    }
}
```