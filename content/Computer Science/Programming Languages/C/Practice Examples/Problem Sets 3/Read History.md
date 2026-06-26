## Task
Implement the function `read_history`
- Open a file in read mode, read all lines, and print each with a line number (starting from 1)
- If `fopen()` fails, call `perror("open")` and return `-1`
- If successful, return `0`

> Format: 
> `line_number command`
> space between number and command

### Function Signature
```c
// Read history file and print each line with line number.
// Format: "1 command"
// Return 0 on success, -1 on error.
int read_history(const char* filepath);
```

### Important File I/O functions
- `fopen(filename, "r")`→ Open file in read mode
- `fgets(buffer, size, file)` → Read line from file
- `fclose(file)` → Close the file

---
## Example
```bash
$ gcc read_history.c -o read_history
$ echo -e "ls -l\npwd\necho hello" > test.txt
$ ./read_history
test.txt
1 ls -l
2 pwd
3 echo hello
$ ./read_history
/nonexistent.txt
open: No such file or directory
```

---
## Code
```c
#include <stdio.h>

int read_history(const char *filepath)
{
    FILE* file = fopen(filepath, "r");
    if(file == NULL){
        perror("open");
        return -1;
    }

    char buffer[100];
    int i = 1;
    while(fgets(buffer, sizeof(buffer), file) != NULL){
        printf("%d %s", i, buffer);
        i++;
    }

    fclose(file);
    return 0;
}
```
