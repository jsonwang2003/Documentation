## Task
Implement the function `write_history`
- Opens a file in append mode, write the command to it, and close the file
- If `fopen()` fails, call `perror("open")` and return `-1`
- If successful, return `0`
- Each command should be on its own line in the file

**Input Format**: Each line is `filepath:command`
### Function Signature
```c
// Append command to the history file at filepath.
// Return 0 on success, -1 on error.
int write_history(const char* filepath, const char* command);
```
### Important File I/O Functions
- `fopen(filepath, "a")` → Open file in append mode (creates if doesn't exist)
- `fprintf(file, "%s\n", command)` → Write formatted string to file
- `fclose(file)` → Close the file
---
## Example
```bash
$ gcc write_history.c -o write_history
$ ./write_history
history.txt:ls -l
success
history.txt:pwd
success
history.txt:echo hello
success
$ cat history.txt
ls -l
pwd
echo hello
$ ./write_history
/invalid/path.txt:test
open: No such file or directory
```

---
## Code
```c
#include <stdio.h>

int write_history(const char* filepath, const char* command){
	FILE* file = fopen(filepath, "a");
	if(file == NULL){
		perror("open");
		return -1;
	}
	
	fprintf(file, "%s\n", command);
	fclose(file);
	return 0;
}
```