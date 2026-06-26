## Task
Implement `second_half` function
- Given a `uint32_t` integer type, return the **Least Significant (rightmost)** 16 bits as a `uint16_t`
- Function Signature
```c
// Given a uint32_t, return the Least significant(rightmost) 16 bits as a uint16_t
// For example:
//   0xFFFFFFFF -> 0xFFFF (65535)
//   0x00000000 -> 0x0000 (0)
//   0x00010001 -> 0x0001 (1)
//   0x80008000 -> 0x8000 (32768)
//   0xDEADBEEF -> 0xBEEF (48879)
uint16_t second_half(uint32_t number);
```
---
## Examples
```bash
$ gcc second_half.c -o second_half
$ ./second_half
4294967295
65535
0
0
65537
1
$ ./second_half < small_input.txt
65535
0
1
32768
48879
$ # The next command is how you should create the output files
$ # It will result in a new file with the output from running ./second_half, which
$ # the grader will check for. You can open the files with vim to check the results!
$ ./second_half < small_input.txt > small_result.txt
$ ./second_half < input.txt > result.txt|
```
---
## Code
```c
#include <stdint,h>

uint16_t second_half(uint32_t number){
	return number & 0xFFFF;
}
```

- `&`: Binary **AND** operator, compares the 2 numbers in binary form and return the number corresponding to each digit

| A   | B   | A AND B |
| --- | --- | ------- |
| 0   | 0   | 0       |
| 0   | 1   | 0       |
| 1   | 0   | 0       |
| 1   | 1   | 1       |

