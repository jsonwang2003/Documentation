## Task
Implement the `main` function
- `is_ascii` function is provided
- Accept a single command line argument (string)
- Call `is_ascii` on that argument
- Print `1` if the string is ASCII, or `0` otherwise
---
## Test
```bash
$ ./is_ascii hello
1
$ ./is_ascii café
0
$ ./is_ascii ASCII
1
$ ./is_ascii 😊
0
```
---
## Code
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
int is_ascii(char* str) {
    while (*str) {
        if ((unsigned char)*str > 127) {
            return 0;
        }
        str++;
    }
    return 1;
}

int main(int argc, char** argv) {
    printf("%d\n", is_ascii(argv[1]));
    return 0;
}
```