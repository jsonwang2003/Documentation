## Task
Implement the function `print_parts`
- The function should print each space-separated part of `str`, one per line, in the format `index: part`
- Consecutive spaces should be treated as a single separator
- Ignore leading and trailing spaces
### Function Signature
```c
void print_parts(const char* str);
```
---
## Example
```bash
# Input
hello   world this   is   c

# Output
0: hello
1: world
2: this
3: is
4: c
```

---
## Code
```c
void print_parts(const char* str) {
    char copy_str[1000];
    int i;
    for(i = 0; str[i] != 0; i++){
        copy_str[i] = str[i];
    }
    copy_str[i] = 0;

    int index = 0;
    char* current = strtok(copy_str, " ");
    while(current != NULL){
        printf("%d: %s\n", index, current);
        index++;
        current = strtok(NULL, " ");
    }
}
```