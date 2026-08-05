## Task
Implement `start_with_10` function
- Given a `char`, returns `1` if it starts with **the bits** `10` and return `0` if it does not
- Function Signature
```c
// Given a char, returns 1 if it starts with the bits `10` and 0 if it does not
// For example:
//   0b10110101 -> 1 (starts with 10)
//   0b11000000 -> 0 (starts with 11)
//   0b01110000 -> 0 (starts with 01)
//   0b10000000 -> 1 (starts with 10)
//   0b00001111 -> 0 (starts with 00)
int starts_with_10(char c);
```
---
## Examples
```bash
$ gcc starts_with_10.c -o starts_with_10
$ ./starts_with_10
181
1
192
0
112
0
128
1
15
0
$ ./starts_with_10 < small_input.txt
1
0
0
1
0
$ # The next command is how you should create the output files
$ # It will result in a new file with the output from running ./starts_with_10, which
$ # the grader will check for. You can open the files with vim to check the results!
$ ./starts_with_10 < small_input.txt > small_result.txt
$ ./starts_with_10 < input.txt > result.txt
```
---
## Code
```c
int starts_with_10(char c){
	return ((c & 0xC0) == 0x80) ? 1 : 0;
}
```

1. Mask the more significant 2 bits and use the `AND` binary operation
2. Check if the masked value is the same only have `10` in its most significant bit (`0x80`)