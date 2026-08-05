## Task
Implement `first_half` function
- Given a `uint32_t`, return the **most significant (leftmost)** 16 bits as a `uint16_t`
- Function Template
```c
// Given a uint32_t, returns the most significant (leftmost) 16 bits as a uint16_t
// For example:
//   0xFFFFFFFF -> 0xFFFF (65535)
//   0x00000000 -> 0x0000 (0)
//   0x00010000 -> 0x0001 (1)
//   0x80000000 -> 0x8000 (32768)
//   0xDEADBEEF -> 0xDEAD (57005)
uint16_t first_half(uint32_t number);
```
---
## Example
```bash
$ gcc first_half.c -o first_half
$ ./first_half
4294967295
65535
0
0
65536
1
$ ./first_half < small_input.txt
65535
0
1
32768
57005
$ # The next command is how you should create the output files
$ # It will result in a new file with the output from running ./first_half, which
$ # the grader will check for. You can open the files with vim to check the results!
$ ./first_half < small_input.txt > small_result.txt
$ ./first_half < input.txt > result.txt
```
---
## Code
```c
#include <stdint.h>

uint16_t first_half(uint32_t number){
	uint32_t result = number & 0xFFFF0000;
	return result >> 16;
}
```