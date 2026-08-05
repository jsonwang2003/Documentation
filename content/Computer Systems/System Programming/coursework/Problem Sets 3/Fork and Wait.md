## Task
Implement `fork_and_run` function
- Fork a child process
	- Child: Execute the given command using `execlp`
	- Parent: wait for the child to complete
- The function takes a command and a single argument
- If fork fails, call `perror("fork")` and return `-1`
- If exec fails in the child, call `perror("exec")` and `exit(1)`
- The parent should wait for the child and return `0` on success
### Important System Calls
- `fork()` → creates a child process, returns `0` in child, child PID in parent, -1 on error
- `execlp(command, command, arg, (char*)NULL)` → replaces process with command
- `wait(NULL)` or `waitpid()` → parent waits for child
### Function Signature
```c
// Fork a child process. Child runs command with arg using execlp.
// Parent waits for child. Return 0 on success, -1 on error.
int fork_and_run(const char* command, const char* arg);
```

---
## Example
```bash
$ gcc fork_and_wait.c -o fork_and_wait
$ ./fork_and_wait
ls -l
total 16
-rwxr-xr-x 1 user user 16704 Nov  6 10:00 fork_and_wait
-rw-r--r-- 1 user user  1234 Nov  6 10:00 fork_and_wait.c
Parent: child completed
$ ./fork_and_wait
echo hello
hello
Parent: child completed
```

---
## Code
```c
#include <sys/wait.h>
#include <unistd.h>

int fork_and_run(const char* command, const char* arg) {
    int pid = fork();
    if (pid < 0){
        perror("fork");
        return -1;
    } else if (pid == 0){
        execlp(command, command, arg, (char*)NULL);
        perror("exec");
        exit(1);
    } else {
        waitpid(pid, NULL, 0);
        return 0;
    }
}
```