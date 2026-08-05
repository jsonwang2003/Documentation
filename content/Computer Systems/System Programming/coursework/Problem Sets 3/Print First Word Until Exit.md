## Task
Implement the function `prompt_command_loop`
- Write a function that repeatedly reads a line from standard input
- For each line, print the **first space-separated word** (or the whole line if there are no spaces)
- Stop the loop and exit the program if the input line is exactly `exit`
- If the program receives `NULL` or `EOF` in the standard input, this should also exit the program
### Function Signature
```c
void prompt_command_loop();
```

---
## Example
```bash
$ ./prompt_command_loop
hello world
hello
foo bar baz
foo
exit
```

---
## Code
```c
void prompt_command_loop(){
    char buffer[100];
    while(fgets(buffer, sizeof(buffer), stdin) != NULL){
        int len = strlen(buffer);
        if (len > 1 && buffer[len - 1] == '\n'){buffer[len - 1] = 0;}

        if (strcmp(buffer, "exit") == 0){break;}

        int i;
        char word[100];
        for(i = 0; buffer[i] == ' '; i++){
            word[i] = buffer[i];
        }
        while(buffer[i] != ' ' && buffer[i] != 0){
            word[i] = buffer[i];
            i++;
        }
        word[i] = 0;

        printf("%s\n", word);
    }
};
```