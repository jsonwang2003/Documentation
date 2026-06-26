## Task
Implement the function `bitwise_is_even`
- Given an **8-bit integer**, return `1` if the number is **even** or `0` if the number is **odd**
- Only using **bitwise operators**

### Function Signature
```c
// Given a int8_t, return 1 if the number is even or 0 if the number is odd only using bitwise operators
// For example:
// Input -> Output
// 0 -> 1
// 1 -> 0
// -2 -> 1
// 29 -> 0
int bitwise_is_even(int8_t n);
```
---
## Examples
```bash
$ gcc bitwise_is_even.c -o bitwise_is_even
$ ./bitwise_is_even
123
0
-29
0
52
1
0
1
$ ./bitwise_is_even < small_input.txt
1
0
1
0
1
0
```
---
## Code
```c
#include <stdint.h>

int bitwise_is_even(int8_t n){
	if(n & 0x01 == 1){
		return 0;
	}
	return 1;
}
```