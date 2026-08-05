## Task
Implement an HTTP server that echoes back the entire request it receives.
- Use the provided `http-server.h` library
- Implement the `handle_request` function
- Echo back the full HTTP request in the response body
### Function Signature
```c
void handle_request(char *request, int client_socket);
```
### HTTP Response Format
```c
HTTP/1.1 200 OK\r\n
Content-Type: text/plain\r\n
\r\n
<the full request echoed here>
```

> [!IMPORTANT]
> Use `\r\n` (carriage return + newline) for line endings in HTTP, not just `\n`.
### Notes
- Use `send(client_socket, data, length, 0)` to send data to the client
- Use `strlen()` to get string lengths
- Send the HTTP header first, then echo the request
- Include `<string.h>` for `strlen()`

---
## Example
If a client sends:
```bash
GET /hello HTTP/1.1
Host: localhost:8000
User-Agent: curl/7.68.0
Accept: */*
```

**Your server should respond with:**

```bash
HTTP/1.1 200 OK
Content-Type: text/plain

GET /hello HTTP/1.1
Host: localhost:8000
User-Agent: curl/7.68.0
Accept: */*
```
### Test Locally
```bash
$ gcc -o echo_server echo_server.c http-server.c
$ ./echo_server
Server started on port 8000

# In another terminal:
$ curl "http://localhost:8000/hello"
GET /hello HTTP/1.1
Host: localhost:8000
User-Agent: curl/7.68.0
Accept: */*
```

---
## Code
```c
#include "http-server.h"
#include <string.h>

void handle_request(char *request, int client_socket)
{  
    char* header = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n";

    send(client_socket, header, strlen(header), 0);
    send(client_socket, request, strlen(request), 0);
}
```