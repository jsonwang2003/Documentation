## Task
Implement `count_occurences`
- Takes a string and a character, and returns the number of times that character appears in the string
### Function Signature
```c
// Returns the number of times the to_count character appears in str
int count_occurrences(const char* str, char to_count);
```
### Input Format
Each line contains a string followed by a space and then the character to count

---
## Test
```bash
$ gcc count_occurrences.c -o count_occurrences
$ ./count_occurrences
hello l
2
banana a
3
test t
2
test x
0
$ ./count_occurrences < small_input.txt
2
3
2
0
3
```

---
## Code
```c
#include <string.h>

int count_occurences(const char* str, char to_count){
	size_t len = strlen(str);
	int count = 0;
	for(int i = 0; i < len; i++){
		if(str[i] == to_count){
			count++;
		}
	}
	return count;
}
```