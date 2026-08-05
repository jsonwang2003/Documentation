## Task
Implement the function `print_abc_char_and_int_backwards`
- Writing a function that takes a null-terminated `char[]` of characters and prints each character along with its integer (ASCII) value, in backwards order that they appear in the array
- The function signature is
```c
// Given a char[] of characters, print the char representation backwards
// For example:
// Input: abc
// Output:
// c 99
// b 98
// a 97
void print_abc_char_and_int_backwards(char str[]);
```
---
## Testing
### Compiling
- use `gcc` to compile the program and use `./` to run the program
```bash
gcc print_abc_char_and_int_backwards.c -o print_abc_char_and_int_backwards
./print_abc_char_and_int_backwards
```

### Example Uses

- This is how the terminal should look after running these lines
```bash
$ gcc print_abc_char_and_int_backwards.c -o print_abc_char_and_int_backwards
$ ./print_abc_char_and_int_backwards
abc123
3 51
2 50
1 49
c 99
b 98
a 97
$ ./print_abc_char_and_int_backwards < small_input.txt
d 100
c 99
b 98
a 97
? 63
! 33
& 38
```
---
## Code
```c
#include <stdio.h>
#include <string.h>

void print_abc_char_and_int_backwards(char str[]){
	size_t max_len = strlen(str);
	for(int i = max_len - 1; i >= 0; i--){
		printf("%c %d\n", str[i], str[i]);
	}
}
```