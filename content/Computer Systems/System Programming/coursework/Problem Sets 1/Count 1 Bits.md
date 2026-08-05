## Task
Implement the function `count_1_bits`
- Given a **32-bit integer**, returns the number of `1` bits only using **bitwise operators**
### Function Signature
```c
// Given a 32-bit integer, return the number of 1 bits (aka the popcount/hamming weight) only using bitwise operators
// For example:
// Input -> Output
// 128 -> 1
//-1 -> 32
// 0 -> 0
// 2147483647 -> 31
int count_1_bits(int32_t n);
```
---
## Example
```bash
$ gcc count_1_bits.c -o count_1_bits
$ ./count_1_bits
123
6
-29
29
999999999
21
$ ./count_1_bits < small_input.txt
1
32
0
1
25
31
```
---
## Code
```c
#include <stdint.h>

int count_1_bits(int32_t n){
	int count = 0;
	for (int i = 0; i < 32; i++, n = n >> 1){
		if (n & 0x00000001 == 1){
			count++;
		}
	}
	return count;
}
```