## Task
Implement a simple HTTP server that responds with "Hello, World!" to any request.
- Use the provided `http-server.h` library
- Implement the `handle_request` function
- Always respond with "Hello, World!" regardless of the request
### Function Signature
```c
void handle_request(char *request, int client_socket);
```
### HTTP Response Format
A minimal HTTP response consists of:
```c
HTTP/1.1 200 OK\r\n
Content-Type: text/plain\r\n
\r\n
Hello, World!
```

> [!Important] 
> Use `\r\n` (carriage return + newline) for line endings in HTTP, not just `\n`.
### Note
- Use `send(client_socket, data, length, 0)` to send data to the client
- Use `strlen()` to get string lengths
- You can build the response as one string or send it in pieces
- Include `<string.h>` for `strlen()`

---
## Example
For any request (e.g., `GET /hello HTTP/1.1` or `GET /test HTTP/1.1`):

**Your server should respond with:**
```bash
HTTP/1.1 200 OK
Content-Type: text/plain

Hello, World!
```
### Test Locally
```bash
$ gcc -o hello_world_server hello_world_server.c http-server.c
$ ./hello_world_server
Server started on port 8000

# In another terminal:
$ curl "http://localhost:8000/hello"
Hello, World!
$ curl "http://localhost:8000/anything"
Hello, World!
```

---
## Code
```c
#include "http-server.h"
#include <string.h>

void handle_request(char *request, int client_socket)
{
    char* data = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nHello, World!";
    send(client_socket, data, strlen(data), 0);
}
```