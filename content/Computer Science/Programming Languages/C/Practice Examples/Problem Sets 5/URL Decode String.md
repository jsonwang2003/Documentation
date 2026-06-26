## Task
Implement the function `url_decode`.
- Write a function that decodes URL-encoded strings in-place
- Convert `%20` to space, `%21` to `!`, etc.
- The general pattern is `%HH` where `HH` is a two-digit hexadecimal number
- Modify the string in-place and return the same pointer
### Function Signature
```c
// Decode URL-encoded string in-place, return the same pointer
char* url_decode(char* str);
```
### Note
- Find all occurrences of `%` followed by two hexadecimal digits
- Convert the hex digits to their ASCII character value
- Replace the three characters `%HH` with the single decoded character
- Shift remaining characters left to fill the gap
- Common encodings: `%20` = space, `%21` = `!`, `%2C` = `,`, `%3F` = `?`
- You can use the function `strtol` to convert hex strings to integers
- Modify the string in-place (don't allocate new memory)
- If you encounter a `%` not followed by two hex digits, you can leave it as-is

---
## Example
|Input|Output|
|---|---|
|`hello%20world`|`hello world`|
|`hi%21`|`hi!`|
|`test%20%20double`|`test double`|
|`no_encoding`|`no_encoding`|
|`mix%20of%21both`|`mix of!both`|

---
## Code
```c
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void shift_left(char* str){
    while(str[2] != 0){
        str[0] = str[2];
        str++;
    }
    str[0] = 0;
}

char *url_decode(char *str)
{
    char* current = strstr(str, "%");
    while(current != NULL){
        char hex[3] = {current[1], current[2], 0};
        char* endptr = NULL;
        long val = strtol(hex, &endptr, 16);

        if(endptr){
            *current = (char)val;
            shift_left(current + 1);
            current = strstr(current + 1, "%");
        }
    }
    return str;
}
```