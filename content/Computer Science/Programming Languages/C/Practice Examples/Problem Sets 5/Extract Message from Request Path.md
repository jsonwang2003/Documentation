## Task
Implement the function `get_message`.
- Write a function that extracts the message from a request path
- The path format could be `/post?user=<username>&message=<message>` or `/react?...&message=<message>&id=<id>`
- Return an interior pointer to the message within the original string
- You should modify the string by replacing the `&` after the message with `\0` (if there is one)
- If there is no `message=` parameter, return `NULL`
### Function Signature
```c
// Given a request path, return a pointer to the message (or NULL if not found)
char* get_message(char* request_path);
```

### Note
- Find `message=` in the string (it could be in `/post`, `/react`, or other paths)
- The message starts immediately after `message=`
- The message ends at the `&` character (if present) or end of string
- Replace the `&` with `\0` to terminate the message string (if there is one)
- Return a pointer to the start of the message (an interior pointer into the original string)
- If there is no `message=` in the path, return `NULL`
- If `message=` exists but there's no `&` after it, you can still return the pointer (it will go to end of string)

---
## Example
|Input|Output|
|---|---|
|`/post?user=joe&message=hi`|`hi`|
|`/react?user=aaron&message=👍&id=5`|`👍`|
|`/post?user=alice&message=hello%20world`|`hello%20world`|
|`/chats`|`NULL`|
|`/post?user=bob`|`NULL`|

---
## Code
```c
#include <stdio.h>
#include <string.h>

char *get_message(char *request_path)
{
    char* message = strstr(request_path, "message=");
    if(message != NULL){
        message += 8;
        char* end = strstr(message, "&");
        if(end != NULL){
            *end = 0;
        }
        return message;
    }
    return NULL;
}
```
