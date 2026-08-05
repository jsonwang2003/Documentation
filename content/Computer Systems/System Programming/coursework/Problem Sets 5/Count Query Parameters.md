## Task
Implement the function `count_parameters`.
- Write a function that counts the number of query parameters in a URL path
- Query parameters are in the format `key=value` and are separated by `&`
- The query string starts after the `?` character
- Return the count of parameters
### Function Signature
```c
// Count the number of query parameters in a URL path
// Returns the count (0 if no parameters)
int count_parameters(char* path);
```

### Note
- If there's no `?` in the path, there are 0 parameters
- Count the number of `=` signs after the `?` to determine parameter count
- Alternatively, count `&` signs and add 1 (if there's at least one parameter)
- You don't need to validate the format of the parameters
- Don't modify the input string

---
## Example
| Input                               | Output |
| ----------------------------------- | ------ |
| `/post?user=joe&message=hi`         | `2`    |
| `/react?user=aaron&message=👍&id=5` | `3`    |
| `/chats`                            | `0`    |
| `/post?user=alice`                  | `1`    |
| `/test?a=1&b=2&c=3&d=4`             | `4`    |

---
## Code
```c
#include <stddef.h>
#include <stdio.h>
#include <string.h>

int count_parameters(char *path)
{
    int count = 0;
    char* start = strstr(path, "?");
    while(start != NULL){
        start = strstr(start + 1, "=");
        if(start != NULL){count++;}
    }
    return count;
}
```