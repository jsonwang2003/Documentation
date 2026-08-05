## Task
Implement the program `cap_variants.c`
- Read strings from `stdin` in a loop (one per line) and for each string, print all variants where exactly one ASCII letter (a-z or A-Z) has its case flipped
- Only consider ASCII letters (a-z, A-Z). **Non-ASCII character** and **non-alphabetic characters** should not be changed
- Print each variant on its own line
---
## Test
```bash
$ gcc cap_variants.c -o cap_variants
$ ./cap_variants
PepPer
pepPer
PEpPer
PePPer
Pepper
PepPEr
PepPeR
hello
Hello
hEllo
heLlo
helLo
hellO
ABC
aBC
AbC
ABc
```
---
## Code
```c
#include <stdio.h>
#include <string.h>

int main() {
    char buffer[100];
    while(1){
        char *maybe_eof = fgets(buffer, sizeof(buffer), stdin);
        if(maybe_eof == NULL){break;}

        size_t len = strlen(buffer);
        if(len > 0 && buffer[len - 1] == '\n'){buffer[len - 1] = 0;}

        for(int i = 0; buffer[i] != 0; i++){
            if(buffer[i] >= 'A' && buffer[i] <= 'Z'){
                buffer[i] += 'a' - 'A';
                printf("%s\n", buffer);
                buffer[i] -= 'a' - 'A';
            } else if(buffer[i] >= 'a' && buffer[i] <= 'z'){
                buffer[i] -= 'a' - 'A';
                printf("%s\n", buffer);
                buffer[i] += 'a' - 'A';
            }
        }
    }

    return 0;
}
```