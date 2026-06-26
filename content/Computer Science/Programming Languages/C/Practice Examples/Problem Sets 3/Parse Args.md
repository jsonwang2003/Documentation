## Task
Implement the function `parse_args`
- Write a function that prepares a command for `execvp`
- Given a line for a command line `ls -l ./`, replaces all consecutive spaces with null terminators, and puts the addresses of the starts of the command and arguments into the result array, with a `NULL` stored in the index after all the arguments
	- `ls -l ./` → `{"ls", "-l", "./", NULL}`
- Assume that result has enough space for the `len` arguments plus the `NULL` value
	- If more than `len` arguments are in `str`, only include the first `len` arguments
	- The default value will be `len = 3` in `main`
- Do not print anything inside of the `parse_args` function. This is handled in the `main` function
### Function Signature
```c
// Write a function that prepares a command for `execvp`. Given a line for a 
// command like "ls -l ./", replaces all consecutive spaces with null
// terminators, and puts the addresses of the starts of the command and args 
// into the result array, with a NULL stored in the index after all the arguments.
// Assume that result has enough space for len arguments plus the NULL value.
// If more than len arguments are in str, only include the first len arguments.
// By default, len = 3 (initialized in main).
//
// For example:
//
// Input: ls -l ./
// Output: {"ls", "-l", "./", NULL}
//
// Input: echo a b c d e f
// Output: {"echo", "a", "b", NULL}
//
void parse_args(char* str, char* result[], int len);
```

---
## Example
```bash
$ gcc parse_args.c -o parse_args
$ ./parse_args
ls -l ./     
result[0]: 'ls'
result[1]: '-l'
result[2]: './'
result[3]: NULL
echo a b c d e f g
result[0]: 'echo'
result[1]: 'a'
result[2]: 'b'
result[3]: NULL
$ ./parse_args < small_input.txt
result[0]: 'ls'
result[1]: '-l'
result[2]: './'
result[3]: NULL
result[0]: 'echo'
result[1]: 'a'
result[2]: 'b'
result[3]: NULL
result[0]: 'vimtutor'
result[1]: NULL
```

---
## Code
```c
#include <string.h>

void parse_args(char* str, char* result[], int len){
    int i = 0;
    char* current = strtok(str, " ");
    while(i < len){
        result[i] = current;
        i++;
        current = strtok(NULL, " ");
    }
    result[len] = NULL;
    current = NULL;
}
```