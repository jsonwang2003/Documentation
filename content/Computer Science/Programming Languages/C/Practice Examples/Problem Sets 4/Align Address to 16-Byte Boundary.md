## Task
Implement the function `align_me16`
- Write a function that takes a `void*` pointer representing a memory address
- If the address is already aligned to a 16-byte boundary (the address is a multiple of 16), return the original pointer
- if not, return a pointer to the next higher address that is aligned to a 16-byte boundary
### Function Signature
```c
// Given a void* pointer, return the same pointer if it is 16-byte aligned, otherwise return the next higher 16-byte aligned address.
void* align_me16(void* ptr);
```

---
## Example
| Input Address | Output Address |
| ------------- | -------------- |
| 0x1000        | 0x1000         |
| 0x11          | 0x20           |
| 0x2f          | 0x30           |
| 0x12345       | 0x12350        |
| 0x7fffffff    | 0x80000000     |

---
## Code
```c
#include <stdint.h>

void* align_me16(void* ptr){
	uint64_t a = (uint64_t) ptr;
	int dif = a % 16;
	if (dif == 0) {return ptr;}
	return (ptr + 16 - dif);
}
```