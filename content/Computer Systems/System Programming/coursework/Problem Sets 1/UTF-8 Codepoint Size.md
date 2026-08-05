## Task
Implement the function `codepoint_size`
- Returns `1`, `2`, `3`, or `4` depending on the **size of the first codepoint** in the string
- Returns `0` if the **string is empty**
- Returns `-1` if the **first byte** is not a valid UTF-8 start byte

### Function Signature
```c
// Returns 1, 2, 3, or 4 depending on the size of the first codepoint in the string.
// Returns 0 if the string is empty
// Returns -1 if the first byte is not a valid UTF-8 start byte
uint8_t codepoint_size(char string[])
```
---
## Examples
```bash
$ gcc codepoint_size.c -o codepoint_size
$ ./codepoint_size 
hello
1
éclaire
2
h
1
$ ./codepoint_size < small_input.txt
$ # The next command is how you should create the output files
$ # It will result in a new file with the output from running ./codepoint_size, which
$ # the grader will check for. You can open the files with vim to check the results!
$ ./codepoint_size < small_input.txt > small_result.txt
$ ./codepoint_size < input.txt > result.txt
```
---
## Code
```c
#include <stdint.h>

// Returns 1, 2, 3, or 4 depending on the size of the first codepoint in the string.
// Returns 0 if the string is empty
// Returns -1 if the first byte is not a valid UTF-8 start byte
uint8_t codepoint_size(char string[]){
	char c = string[0];
	if(c == 0){return 0;}
	if((c & 0x80) == 0){return 1;}
	if((c & 0xE0) == 0xC0){return 2;}
	if((c & 0xF0) == 0xE0){return 3;}
	if((c & 0xF8) == 0xF0){return 4;}
	return -1;
}
```