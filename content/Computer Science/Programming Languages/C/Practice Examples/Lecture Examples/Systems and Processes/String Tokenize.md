## String Tokenization (`strtok`)

> [!ABSTRACT] 
> **Tokenization** is the process of breaking a large string into smaller chunks (tokens) based on specific separators called **delimiters**. In C, this is commonly handled by `strtok`.

### 1. How `strtok` Works
The `strtok` function maintains internal state to track where it left off in a string.
- **Initial Call**: Pass the string you want to split (e.g., `cmd`). It finds the first delimiter, replaces it with a **Null Terminator (`\0`)**, and returns a pointer to the start of the token.
- **Subsequent Calls**: Pass `NULL` as the first argument. `strtok` uses a global internal pointer to find and return the next token in the same string.
- **Termination**: When it reaches the end of the string (`\0`), it returns `NULL`.

---
### 2. Implementation: `parse_args`
In a shell, we need to convert the user's input into a `char**` (an array of strings) so it can be passed to system calls like `execvp()`.

```c
// Fills result with string elements separated by spaces
// Returns the total number of arguments (argc)
int parse_args(char* cmd, char** result) {
    char* current = strtok(cmd, " "); // Start with the actual string
    int index = 0;
    
    while(current != NULL) {
        result[index] = current;      // Store pointer to the current token
        current = strtok(NULL, " ");  // Move to next token using NULL
        index++;
    }
    return index; // Return count of tokens found
}
```

---
### 3. Integration: The Shell Loop
The shell reads input, cleans it, tokenizes it, and then prepares it for a **Process**.

```c
int main() {
    char cmd[CMD_LENGTH];
    char* args[CMD_LENGTH];
    
    while(1) {
        printf("→ ");
        if(fgets(cmd, sizeof(cmd), stdin) == NULL) break;
        
        // Remove trailing newline added by fgets
        cmd[strcspn(cmd, "\n")] = 0; 
        
        // Convert "cp a.c b.c" into ["cp", "a.c", "b.c"]
        int argc = parse_args(cmd, args);
        
        // Critical for execvp: The argument array must end with NULL
        args[argc] = NULL; 

        // At this point, the shell would fork() and execvp(args[0], args)
    }
}
```

---
### 4. Memory Safety Considerations
- **Destructive Function**: `strtok` modifies the original string by inserting `\0` characters. If you need the original command string later, you must make a copy before tokenizing.
- **Pointer Lifetime**: The pointers in `args[]` point directly into the `cmd[]` buffer. Since `cmd[]` is a local **Stack** variable in `main`, these pointers remain valid as long as `main` is running.
### Module Navigation
This concludes the fundamental concepts for your **Systems Programming & Memory** chapter.
- **[[Process of Operating System]]**: Review how these `args` are used to launch new programs.
- **[[Pointers and Reference]]**: Review why `char**` is used for an array of strings.
- **[[Sizeof]]**: Review how to safely measure buffers like `CMD_LENGTH`.