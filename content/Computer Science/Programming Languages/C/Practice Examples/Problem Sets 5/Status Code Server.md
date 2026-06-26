## Task
Implement an HTTP server that returns different status codes based on the request path.
- Parse the HTTP request to extract the path
- If the path is `/home`, return `200 OK` with message "Welcome home!"
- For any other path, return `404 Not Found` with message "Not found"
### Function Signature
```c
void handle_request(char *request, int client_socket);
```
### Request Format
An HTTP request line looks like:
```c
GET /home HTTP/1.1
```
The path is between `GET` and the space before `HTTP/1.1` (or before `?` if there are query parameters).
### HTTP Response Format
**For `/home` (200 OK):**
```c
HTTP/1.1 200 OK\r\n
Content-Type: text/plain\r\n
\r\n
Welcome home!
```

**For other paths (404 Not Found):**
```c
HTTP/1.1 404 Not Found\r\n
Content-Type: text/plain\r\n
\r\n
Not found
```
### Notes
- Use `strstr()` to find `"GET "` in the request
- Use `strncmp()` to check if the path matches `/home`
- The path must be **exactly** `/home` (not `/home/page` or `/homepage`)
- Use `send()` to send responses
- Include `<string.h>` for string functions
---
## Example
|Request Path|Status Code|Response Body|
|---|---|---|
|`GET /home HTTP/1.1`|200 OK|`Welcome home!`|
|`GET /hello HTTP/1.1`|404 Not Found|`Not found`|
|`GET /test HTTP/1.1`|404 Not Found|`Not found`|
|`GET / HTTP/1.1`|404 Not Found|`Not found`|
### Test Locally
```bash
$ gcc -o status_server status_server.c http-server.c
$ ./status_server
Server started on port 8000

# In another terminal:
$ curl "http://localhost:8000/home"
Welcome home!

$ curl "http://localhost:8000/hello"
Not found

$ curl -i "http://localhost:8000/home"
HTTP/1.1 200 OK
Content-Type: text/plain

Welcome home!

$ curl -i "http://localhost:8000/hello"
HTTP/1.1 404 Not Found
Content-Type: text/plain

Not found
```

---
## Code
```c
#include "http-server.h"
#include <string.h>

void handle_request(char *request, int client_socket){
    char* start = strstr(request, "GET ") + 4;
    char* end = strstr(start, " ");
    if(end == NULL){
        char* end = strstr(start, "?");
    }
    int len = end - start;
    if(strncmp(start, "/home", 5) == 0){
        char* header = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nWelcome home!";
        send(client_socket, header, strlen(header), 0);
    } else {
        char* header = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\nNot found";
        send(client_socket, header, strlen(header), 0);
    }
}
```