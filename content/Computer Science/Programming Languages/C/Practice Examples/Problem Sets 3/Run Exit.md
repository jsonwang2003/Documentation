## Task
Implement the function `run_exit`
- Check if the `exit` command is called correctly (with no arguments), then call `exit(EXIT_SUCCESS)` to terminate the program
- If arguments are provided, print `usage: exit` to `stderr` and return `-1` (**do NOT** exit)
- Takes `argc` (number of arguments) and return `0` on success, `-1` on error

> [!IMPORTANT]
> When the exit command is correct, your function should call `exit(EXIT_SUCCESS)` and never return
### Function Signature
```c
// Handle the exit command. Should have argc == 1 (just "exit", no args).
// If correct, call exit(EXIT_SUCCESS). If wrong, print usage error and return -1.
int run_exit(int argc);
```

---
## Example
```bash
$ gcc run_exit.c -o run_exit
$ ./run_exit
exit
$ echo $?
0
$ ./run_exit
exit arg1
usage: exit
$ ./run_exit
exit
<program exits immediately>
```

---
## Code
```c
#include <stdlib.h>
#include <stdio.h>

int run_exit(int argc)
{
    if(argc != 1){
        fprintf(stderr, "usage: exit");
        return -1;
    }
    exit(EXIT_SUCCESS);
}
```