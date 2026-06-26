## Tasks
Implement the function `count_0_bits`
- Given an 32-bit integer, return the number of 0 bits in it
### Function Signature
```c
// Given a 32-bit integer, return the number of 0 bits
// For example:
// Input -> Output
// 128 -> 31  (has one 1 bit, so thirty-one 0 bits)
// -1 -> 0    (all bits are 1, so zero 0 bits)
// 0 -> 32    (all bits are 0)
// 2147483647 -> 1 (all bits except the sign bit are 1)
int count_0_bits(int32_t n);
```
---
## Testing
```bash
$ gcc count_0_bits.c -o count_0_bits
$ ./count_0_bits
123
26
-29
3
999999999
11
$ ./count_0_bits < small_input.txt
31
0
32
31
7
1
```
---
## Code
```c
#include <stdint.h>

int count_0_bits(int32_t n){
	int count = 0;
	for(int i = 0; i < 32; i++){
		if((n & 0x00000001) == 0){
			count++;
		}
		n = n >> 1;
	}
	return count;
}
```