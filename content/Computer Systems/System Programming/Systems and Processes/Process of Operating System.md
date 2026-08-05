> [!ABSTRACT]
> 
> A Process is a running instance of a program. This section covers how the Operating System (OS) manages these instances through unique identifiers, isolated memory regions, and specific system calls for process creation and replacement.

---
### 1. Anatomy of a Process
When you run a command like `./pwcrack`, the OS creates a process. Every process is defined by two main components:
- **PID (Process ID)**: A unique integer used by the OS to track and manage the process.
- **Address Space**: The isolated "sandbox" of memory assigned to the process.
#### Memory Stack Organization (The Address Space)
Within the address space, memory is partitioned into specific regions managed by the Operating System. These sections are tracked by specific CPU registers to ensure the program executes correctly:
- **Instructions (Code)**: This section contains the read-only compiled instructions of the program. It is pointed to by the **PC (Program Counter)**, which tracks the current instruction being executed.
- **Data (Global Variables)**: This region stores static data and global variables that persist for the entire life of the program. It is tracked by the **AR (Address Register)**.
- **Stack**: This section is used to store local variables and function call frames.
    - **SP (Stack Pointer)**: This register points to the current "top" of the stack.
    - **Direction of Growth**: As indicated by the upward arrow in the diagram, the stack grows **upwards** toward lower memory addresses (e.g., from 3001 toward 2000) as data is added.

![[Pasted image 20251024100720.png]]

---
### 2. Process Control System Calls
In C, we interact with the OS using three primary system calls to manage the "Lifecycle" of a process.

#### `fork()`: The Split
Creates an exact **copy** of the current process. Both processes continue execution from the line immediately following the `fork()` call.
- **In Parent**: Returns the PID of the new child.
- **In Child**: Returns `0`.

![[Pasted image 20251024102115.png]]

#### `execvp()`: The Replacement
Replaces the entire address space (Code, Stack, Global variables) of the current process with a new program.
- The **PID** remains the same.
- The original code is gone; the process "becomes" the new program.

![[Pasted image 20251024103308.png]]

#### `waitpid()`: The Synchronization
Tells a parent process to "pause" until a specific child process finishes.
- `pid`: The child to wait for.
- `status`: Pointer to an integer where exit information is stored (use `NULL` to ignore).

---
### 3. Implementation: A Basic Shell Loop
This code demonstrates the **Fork-Exec-Wait** pattern used by real shells (like `bash` or `zsh`).

```c
#include <unistd.h>
#include <sys/wait.h>
#include <stdio.h>

int main() {
    char cmd[1000];
    char* args[1000];
    
    while(1) {
        printf("→ "); // The shell prompt
        if (fgets(cmd, 1000, stdin) == NULL) break;
        
        // ... (Parsing logic to fill args array) ...

        int pid = fork(); // Create a duplicate process
        
        if (pid == 0) {
            // CHILD: Replace yourself with the command the user typed
            execvp(args[0], args);
            // If execvp returns, it means the command failed
            perror("exec failed");
            return 1;
        } else {
            // PARENT: Wait for the child to finish before showing the prompt again
            printf("Starting process %d\n", pid);
            waitpid(pid, NULL, 0);
        }
    }
}
```

---
### 4. Process Relationships Summary

|**System Call**|**Effect on Memory**|**Effect on PID**|
|---|---|---|
|`fork()`|Duplicates entire address space|New PID for child|
|`execvp()`|Overwrites address space with new code|Keeps existing PID|
|`waitpid()`|None (Pauses execution)|None|





```bash
$ ./pwcrack 123abc...
```

> Operating Starts "**Process**": A running program

![[Pasted image 20251024100720.png]]

### Vocabulary:
- **PID (process id)**: unique id the *operating system* uses to track processes
- **Address Space**: The memory given to process by the *operating system*
- **Section (of address space)**
	- Operating system
	- Global Variables
	- Code
	- Stack
---
## System Calls
- `int fork()`: Makes a **copy** of the current process
	- new process starts running immediately
	- starts running from the same line in the code (the line right after `fork()`)
	- The "**parent**" process also keeps running!

> [!IMPORTANT]
> Returns `0` in the child process
> Returns `pid` of the child process in parent
> ```c
> int pid = fork();
> if(pid == 0) { 
> 	// child pcrocess
> } else { 
> 	// parent process continues
> }
> ```

![[Pasted image 20251024102115.png]]

- `void execvp(char* program, char** args)`: Replaces the **current process** with a newly-started process based on the given `cmd` and `args`
	- Similar to **started from command line**
	- Keeps `pid`, `stdin`, `stdout`
	- All the code, variables, and data are **overridden**

![[Pasted image 20251024103308.png]]

- `void waitpid(int pid, int* status, int options)`: Wait for the subprogram to finish before resuming the parent process
	- `pid`: the process id that want to wait until finish
	- `status`: 
		- `NULL` for not caring how it terminated
	- `options`: 
		- `0`: default option

---

```c
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

#define CMD_LENGTH 1000

// Puts arguments into result using strtok
int parse_args(char* cmd, char** result) {/* run strtok*/}

int main(){
	char cmd[CMD_LENGTH];
	char* args[CMD_LENGTH];
	
	while(1){
		printf("→ ");
		char* result = fgets(cmd, CMD_LENGTH, stdin);
		if(result == NULL){ break; }
		cmd[strcspn(cmd, "\n")] = 0;
		
		int argc = parse_args(cmd, args);
		args[argc] = NULL;
		
		int pid = fork();
		
		if(pid < 0){
			printf("fork() failed: %d\n", pid);
			break;
		} else if(pid == 0){
			// child process
			execvp(args[0], args);
		} else {
			// parent process (the shell)
			printf("Starting %d\n", pid);
			waitpid(pid, NULL, 0);
		}
	}
}
```