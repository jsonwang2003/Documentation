## Task
Implementation `lowest_n` as described below
- Given a **32-bit unsigned** integer`x` and an **unsigned 8-bit** integer `n`, return an **unsigned 32-bit** integer containing **just the least `n` significant** bits of `x`
- if `n >= 32` → return `x` unchanged
- if `n == 0` → return 0
### Function Signature
```c
// Given a 32-bit unsigned integer x and an unsigned 8-bit integer n,
// return an unsigned 32-bit integer containing just the least n significant bits of x.
// If n is greater than or equal to 32, return x unchanged.
// If n is 0, return 0.
// For example:
//   x = 0xABCDEF12, n = 4 produces 0x00000002 (keeps only last 4 bits: 0010)
//   x = 0x89ABCDEF, n = 12 produces 0x00000DEF (keeps last 12 bits)
//   x = 0xFFFFFFFF, n = 0 produces 0x00000000 (keeps no bits)
//   x = 0x12345678, n = 40 produces 0x12345678 (n >= 32, returns original)
uint32_t lowest_n(uint32_t x, uint8_t n);
```
---
## Examples
```bash
$ gcc lowest_n.c -o lowest_n
$ ./lowest_n
0xABCDEF12 4
0x00000002
0x89ABCDEF 12
0x00000DEF
0xFFFFFFFF 0
0x00000000
$ ./lowest_n < small_input.txt
0x00000002
0x00000DEF
0x00000000
0x12345678
$ # The next command is how you should create the output files
$ # It will result in a new file with the output from running ./lowest_n, which
$ # the grader will check for. You can open the files with vim to check the results!
$ ./lowest_n < small_input.txt > small_result.txt
$ ./lowest_n < input.txt > result.txt
```
---
## Code
```c
#include <stdint.h>

uint32_t lowest_n(uint32_t x, uint8_t n){
	if(n >= 32){ return x; }
	if(n == 0) {return 0; }
	uint32_t result = x << (32 - n);
	return (result >> (32 - n));
}
```