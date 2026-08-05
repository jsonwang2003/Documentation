## Task
Implement the function `print_abc_char_and_int`
- Write a function that takes a null-terminated `char[]` of characters and prints each character along with its integer (ASCII) value
- Function Signature
```c
// Given a char[] of characters, print the char representation and int representation
// For example:
// Input: "abc"
// Output: 
// a 97
// b 98
// c 99
void print_abc_char_and_int(char str[]);
```
---
## Testing
### Compiling
- Use `gcc` to compile the program and use `./` to run the program
```bash
gcc print_abc_char_and_int.c -o print_abc_char_and_int
./print_abc_char_and_int
```
### Example Uses
- This is how the terminal should look after running these commands
```bash
gcc print_abc_char_and_int.c -o print_abc_char_and_int
./print_abc_char_and_int
abc123
a 97
b 98
c 99
1 49
2 50
3 51
./print_abc_char_and_int < small_input.txt
a 97
b 98
c 99
d 100
& 38
! 33
? 63
```

---
## Code
```c
#include <stdio.h>
#include <string.h>

void print_abc_char_and_int(char str[]){
	int i = 0;
	while(str[i] != '\0'){
		printf("%c %d\n", str[i], str[i]);
		i++;
	}
}
```

