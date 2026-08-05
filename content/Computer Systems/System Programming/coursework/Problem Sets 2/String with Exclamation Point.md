## Task
Implement `main.c` file with a `main` function that reads a **typed line of input** form the user and prints that string followed by an **exclamation point**
- If the user enters more than 20 bytes, it should include only the first 20 bytes of what they typed
---
## Test
```bash
$ gcc main.c -o main
$ ./main
hello
hello
!
$ ./main
this is really long and more than 20 chars
this is really long !
```
---
## Code
```c
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv){
    char buffer[100];
    while(1){
        char* maybe_eof = fgets(buffer, sizeof(buffer), stdin);
        if(maybe_eof == NULL){break;}

        size_t len = strlen(buffer);
        if (len > 20){
            buffer[20] = '!';
            buffer[21] = 0;
        } else {
            buffer[len] = '!';
        }

        printf("%s\n", buffer);
    }
    return 0;
}
```