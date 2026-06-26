## Task
Implement the function `get_id`.
- Write a function that extracts the ID from a react request path
- The path format is `/react?user=<username>&message=<reaction>&id=<id>`
- Return an interior pointer to the ID string within the original string
- You should modify the string by replacing any character after the ID (space, `&`, etc.) with `\0`
- If there is no `id=` parameter, return `NULL`
### Function Signature
```c
// Given a react request path, return a pointer to the ID string (or NULL if not found)
char* get_id(char* react_request);
```

### Notes
- Find `id=` in the string
- The ID starts immediately after `id=`
- The ID ends at a space, `&`, or end of string
- Replace the terminating character with `\0` (if it's not already end of string)
- Return a pointer to the start of the ID (an interior pointer into the original string)
- If there is no `id=` in the path, return `NULL`
- The ID is returned as a string (you don't need to convert to integer)
- `strstr` may be a helpful function to find substrings

---
## Example
|Input|Output|
|---|---|
|`/react?user=aaron&message=👍&id=5`|`5`|
|`/react?user=joe&message=cool&id=123`|`123`|
|`/react?user=alice&id=7&message=nice`|`7`|
|`/react?user=bob&message=wow&id=42 HTTP/1.1`|`42`|
|`/post?user=joe&message=hi`|`NULL`|

---
## Code
```c
#include <stddef.h>
#include <stdio.h>
#include <string.h>

char* get_id(char* react_request)
{
    char* id = strstr(react_request, "id=");
    if(id == NULL){return NULL;}
    char* end = strstr(id + 3, " ");
    if (end == NULL){
        end = strstr(id + 3, "&");
    }
    if (end != NULL){
        *end = 0;
    }
    return id + 3;
}
```