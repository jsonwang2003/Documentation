## Task
Implement the function `run_ls`
- Takes a `char* path` to a directory and runs `ls` on that path using one of the exec family of functions
- Should completely replace the current process (must not return)
- If `exec` fails, you should print an error using `perror`
### Function Signature
```c
// Given a path, execute `ls` on that path, replacing the current process.
void run_ls(char* path);
```
### Note
- Use `fgets` or similar to read the path string from `stdin`
- Strip `'\n'` if present
- May use `execlp`, `execl`, or any other `exec*` function
- If the `exec` call fails, you should print a meaningful error and `exit(1)`
- Make sure function does not return if the exec call is successful

---
## Example
```bash
$ ./run_ls
.
run_ls
run_ls.c

$ ./run_ls
/nonexistent
ls: cannot access '/nonexistent': No such file or directory
```

---
## Code
```c
int run_ls(const char str[]){
    execlp("ls", "ls", str, (char*)NULL);
    perror("exec");
    exit(1);
}
```