## Task
Implement the function `get_username`.
- Write a function that extracts the username from a post request path
- The path format is `/post?user=<username>&message=<message>`
- Return an interior pointer to the username within the original string
- You should modify the string by replacing the `&` after the username with `\0`
### Function Signature
```c
// Given a /post request path, return a pointer to the username
char* get_username(char* post_request);
```

### Note
- Find `user=` in the string
- The username starts immediately after `user=`
- The username ends at the `&` character
- Replace the `&` with `\0` to terminate the username string
- Return a pointer to the start of the username (an interior pointer into the original string)
- You may assume the input always has the format `/post?user=<username>&message=<message>`

---
## Example
|Input|Output|
|---|---|
|`/post?user=joe&message=hi`|`joe`|
|`/post?user=aaron&message=sup`|`aaron`|
|`/post?user=alice&message=hello%20world`|`alice`|
|`/post?user=bob&message=test`|`bob`|
|`/post?user=x&message=short`|`x`|

---
## Code
```c
#include <string.h>
#include <stdio.h>

char *get_username(char *post_request)
{
    char* user = strstr(post_request, "user=") + 5;
    char* end = strstr(user, "&");
    *end = 0;
    return user;
}
```