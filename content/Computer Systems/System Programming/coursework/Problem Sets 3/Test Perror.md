## Task
Implement `test_fopen` and `test_chdir`
- `test_fopen`
	- Try to open a file
	- If it fails, call `perror("open")` and return `-1`
	- If success, close it and return `0`
- `test_chdir`
	- Try to change directory
	- If it fails, call `perror("cd")` and return `-1`
	- If success, return `0`
### Function Signature
```c
// Try to open filepath. If fopen fails, call perror("open") and return -1.
// If success, close file and return 0.
int test_fopen(const char* filepath);

// Try to chdir to path. If chdir fails, call perror("cd") and return -1.
// If success, return 0.
int test_chdir(const char* path);
```
### What is `perror()`
```c
void perror(const char* s)
```

- Produces a message on `stderr` describing the last error encountered during a library function / system call
- When printing the error message, `perror()` prints in this format:
```bash
<s>: <error message>
```
- Consider an example where we try to open a nonexistent file using `fopen()`, the call should fail and not return a valid `FILE` pointer, in which case, we call `perror()` to report the problem
`perror("open")` prints `open: No such file or directory`

> [!NOTE]
> Use `man` or other resources to see what `fopen` and `chdir` will return upon error

---
## Example
```bash
$ gcc test_perror.c -o test_perror
$ ./test_perror
open /nonexistent.txt
open: No such file or directory
open test_perror.c
success
cd /nonexistent_dir
cd: No such file or directory
cd .
success
```

---
## Code
```c
int test_fopen(const char *filepath)
{
    FILE *file;
    file = fopen(filepath, "r");

    if(file == NULL){
        perror("open");
        return -1;
    }

    fclose(file);
    return 0;
}

int test_chdir(const char *path)
{
    if(chdir(path) == 0){ return 0;}
    else{
        perror("cd");
        return -1;
    }
}
```