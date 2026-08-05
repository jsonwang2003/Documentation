## Task
Implement the function `is_aligned16`
- Write a function that takes a `void*` pointer representing a memory address
- Return `1` if the address is aligned to a 16-byte boundary, `0` otherwise
	- The address is a multiple of 16
### Function Signature
```c
// Given a void* pointer, return 1 if the address is aligned to 16 bytes, else 0.
int is_aligned16(void* ptr);
```
---
## Example
```bash
0x0 1
0x11 0
0x20 1
0xff 0
```
---
## Code
```c
#include <stdint.h>

int is_aligned16(void* ptr){
	uint64_t a = (uint64_t) ptr;
	return (a % 16 == 0) ? 1 : 0;
}
```