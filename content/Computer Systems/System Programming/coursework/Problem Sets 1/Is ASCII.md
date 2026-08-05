## Task
Implement the function `is_ascii`
- Returns `1` if the given string contains only **ASCII** characters (or is empty)
- Return `0` if there are any **non-ASCII** characters in the string
### Function Signature
```c
// Returns `1` if the given string contains only ASCII characters (or
// is empty), and `0` if there are any non-ASCII characters in the string
uint8_t is_ascii(char str[]);
```
---
## Examples
```bash
$ gcc is_ascii.c -o is_ascii
$ ./is_ascii 
hello
1
éclaire
0
$ ./is_ascii < small_input.txt
1
0
0
0
1
$ # The next command is how you should create the output files
$ # It will result in a new file with the output from running ./is_ascii, which
$ # the grader will check for
$ ./is_ascii < small_input.txt > small_result.txt
$ ./is_ascii < input.txt > result.txt
```
---
## Code
```c
#include <stdint.h>

uint8_t is_ascii(char string[]){
	for(int i = 0; string[i] != 0; i++){
		if((string[i] & 0x80) == 0x80){return 0;}
	}
	return 1;
}
```