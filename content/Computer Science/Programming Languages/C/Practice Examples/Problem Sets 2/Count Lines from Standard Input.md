## Task
Implement a program that 
- Reads from `stdin` in a loop util EOF (end of file) and counts the total number of lines
- After reaching EOF, print the **total count** as a single number

> Use a loop with `fgets()` to read lines until it returns `NULL` (indicating EOF)
---
## Test
```bash
$ gcc count_lines.c -o count_lines
$ ./count_lines
hello
world
test
<Press Ctrl-D for end of input>
3
$ ./count_lines
one line
<Press Ctrl-D for end of input>
1
$ # The next command is how you should create the output files
$ # It will result in a new file with the output from running ./count_lines, which
$ # the grader will check for
$ ./count_lines < small_input.txt > small_result.txt
$ ./count_lines < input.txt > result.txt
```
---
## Code
```c
#include <stdio.h>
#include <string.h>

int main(){
	int count = 0;
	char buffer[100];
	while(1){
		char* maybe_eof = fgets(buffer, sizeof(buffer), stdin);
		if(maybe_eof == NULL){
			break;
		}
		count++;
	}
	printf("%d\n", count);
	
	return 0;
}
```