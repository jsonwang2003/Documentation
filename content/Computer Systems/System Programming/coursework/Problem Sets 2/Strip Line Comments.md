## Task
Implement `strip_line_comments`
- Remove everything from `//` to the end of the line.
- If there's no `//`, return the string unchanged
### Function Signature
```c
// Removes everything from "//" to the end of the string (in-place).
// If no "//" is found, the string is unchanged.
void strip_line_comments(char str[]);
```
---
## Test
```bash
$ gcc strip_line_comments.c -o strip_line_comments
$ ./strip_line_comments 
int x = 5; // this is a comment
int x = 5; 
// full line comment

int y = 10;
int y = 10;
x = 5; // comment // more
x = 5; 
$ ./strip_line_comments < small_input.txt
int x = 5; 

int y = 10;
x = 5; 
code 
test
```
---
## Code
```c
void strip_line_comments(char str[]) {
    int i = 0;
    while(str[i] != 0){
        if (str[i] == '/' && str[i + 1] == '/'){
            str[i] = 0;
        }
        i++;
    }
}
```