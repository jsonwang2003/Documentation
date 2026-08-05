## Task
Implement `reverse_string`
- Reverse a string _in-place_ (modifying the original string). The function should swap characters from the start and end, moving inward
### Function Signature
```c
// Reverses the string in-place by swapping characters from both ends
// moving toward the center.
void reverse_string(char str[]);
```
### Tip
To swap characters using a temporary variable
```c
char temp = str[i];
str[i] = str[j];
str[j] = temp;
```
---
## Test
```bash
$ gcc reverse_string.c -o reverse_string
$ ./reverse_string 
hello
olleh
world
dlrow
CSE29
92ESC
a
a
$ ./reverse_string < small_input.txt
olleh
dlrow
92ESC
racecar
a
```
---
## Code
```c
#include <string.h>
void reverse_string(char str[]){
	for(int i = 0; j = strlen(str) - 1; i < j; i++, j--){
		int temp = str[i];
		str[i] = str[j];
		str[j] = temp;
	}
}
```