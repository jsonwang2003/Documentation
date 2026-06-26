## Task
Implement the function `recognize_path`.
- Write a function that takes an HTTP GET request string and returns the path type.
- If the path is `/post`, return the string literal `"post"`
- If the path is `/react`, return the string literal `"react"`
- If the path is `/chats`, return the string literal `"chats"`
- If the path is anything else, return `NULL`
### Function Signature
```c
// Given an HTTP GET request line, return "post", "react", "chats", or NULL
char* recognize_path(char* request);
```
### Notes
- The path starts after `GET` and ends at either `?` or a space
- Ignore any query parameters (everything after `?`)
- Return the exact string literals `"post"`, `"react"`, or `"chats"` (not dynamically allocated)
- The path must match exactly (case-sensitive)

---
## Example
| Request Line                                     | Output  |
| ------------------------------------------------ | ------- |
| `GET /post?user=joe&message=hi HTTP/1.1`         | `post`  |
| `GET /react?user=aaron&message=👍&id=5 HTTP/1.1` | `react` |
| `GET /chats HTTP/1.1`                            | `chats` |
| `GET /unknown HTTP/1.1`                          | `NULL`  |
| `GET /reset HTTP/1.1`                            | `NULL`  |

---
## Code
```c
#include <string.h>
#include <stdio.h>

char *recognize_path(char *request)
{
    char POST_PATH[] = "GET /post";
    char REACT_PATH[] = "GET /react";
    char CHATS_PATH[] = "GET /chats";
    if(strncmp(request, POST_PATH, strlen(POST_PATH)) == 0 
                    && (request[strlen(POST_PATH)] == '?' 
                    || request[strlen(POST_PATH)] == ' ')){
        return "post";
    } else if(strncmp(request, REACT_PATH, strlen(REACT_PATH)) == 0
                    && (request[strlen(REACT_PATH)] == '?' 
                    || request[strlen(REACT_PATH)] == ' ')){
        return "react";
    } else if(strncmp(request, CHATS_PATH, strlen(CHATS_PATH)) == 0
                    && (request[strlen(CHATS_PATH)] == '?' 
                    || request[strlen(CHATS_PATH)] == ' ')){
        return "chats";
    }
    return NULL;
}
```