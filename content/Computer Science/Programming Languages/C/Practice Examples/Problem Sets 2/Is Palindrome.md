## Task
Implement `is_palindrome`
- Check if a string is a palindrome (reads the same forwards and backwards). Return `1` if it is a palindrome or `0` if it is not
### Function Signature
```c
// Returns 1 if the string is a palindrome (reads the same forwards 
// and backwards), 0 otherwise.
uint8_t is_palindrome(char str[]);
```
---
## Test
```bash
$ gcc is_palindrome.c -o is_palindrome
$ ./is_palindrome 
racecar
1
hello
0
madam
1
noon
1
test
0
$ ./is_palindrome < small_input.txt
1
0
1
1
0
```
---
## Code
```c
#include <stdint.h>
#include <string.h>

uint8_t is_palindrome(char str[]){
	for(int i = 0, j = strlen(str) - 1; i < j; i++, j--){
		if(str[i] != str[j]){
			return 0;
		}
	}
	return 1;
}
```