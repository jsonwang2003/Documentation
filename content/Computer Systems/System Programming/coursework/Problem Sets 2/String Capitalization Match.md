## Task
Implement the function `string_cap_match`
- The program reads lines from `stdin`. Each line contains two strings separated by a space. For each pair of strings, your function should determine if they are:
	- Identical
	- Differ by exactly one character's capitalization
- Take two strings as parameters
- Return `1` if the strings match / differ by exactly one capitalization change
- Return `0` otherwise

### Function Signature
```c
// Returns 1 if strings match or differ by exactly one capitalization change
// Returns 0 otherwise
// Examples:
//   "secret" and "seCret" -> 1 (one change: c->C)
//   "secret" and "seCreT" -> 0 (two changes: c->C and t->T)
//   "hello" and "hello" -> 1 (identical)
//   "World" and "world" -> 1 (one change: W->w)
int string_cap_match(char *s1, char *s2);
```
---
## Test
```shell
$ ./string_cap_match
secret seCret
match
secret seCreT
miss
hello hello
match
World world
match
```
---
## Code
```c
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int string_cap_match(char *s1, char *s2)
{
    if(strlen(s1) != strlen(s2)) {return 0;}
    int count = 0, difference = 'a' - 'A';
    for(int i = 0; s1[i] != 0; i++){
        if(s1[i] != s2[i]){
            if(s1[i] + difference == s2[i] || s1[i] - difference == s2[i]){
                count++;
                if(count >= 2){
                    return 0;
                }
            } else {
                return 0;
            }
        }
    }

    return 1;
}

int main(int argc, char **argv)
{
    char buffer[200];
    while (1)
    {
        char *maybe_eof = fgets(buffer, sizeof(buffer), stdin);
        if(maybe_eof == NULL){ break; }

        size_t len = strlen(buffer);
        if(len > 0 && buffer[len - 1] == '\n'){
            buffer[len - 1] = 0;
        }

        char* space_pos = strchr(buffer, ' ');
        if(space_pos == NULL){ continue; }

        *space_pos = '\0';
        char *s1 = buffer;
        char *s2 = space_pos + 1;

        printf("%s\n", (string_cap_match(s1, s2)) ? "match" : "miss");
    }

    return 0;
}
```