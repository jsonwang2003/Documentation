## Task
Implement the function `run_cd`
- Check if the `cd` command is called correctly (with exactly 1 argument: the path), then call `chdir()` to change directories
- if `argc != 2`, print `usage: cd <path>` to `stderr` and return `-1`
- if `chdir()` fails, call `perror("cd")` and return `-1`
- If successful, return `0`

```c
// Handle the cd command. Should have argc == 2 ("cd" and path).
// Call chdir() on the path. Return 0 on success, -1 on error.
int run_cd(int argc, char* path);
```

> [!TIP]
> use `chdir(path)` to change directories. It returns `0` on success, `-1` on failure

---
## Example
```bash
$ gcc run_cd.c -o run_cd
$ ./run_cd
cd ..
success
cd
usage: cd <path>
cd /nonexistent
cd: No such file or directory
cd .
success
```

---
## Code
```c
#include <stdio.h>
#include <unistd.h>
int run_cd(int argc, char *path)
{
    if(argc != 2){
        fprintf(stderr, "usage: cd %s\n", path);
        return -1;
    }
    if(chdir(path) < 0){
        perror("cd");
        return -1;
    }
    return 0;
}
```