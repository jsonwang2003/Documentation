## Task
Implement an HTTP server with a calculator endpoint that adds two numbers.
- Parse the HTTP request to extract the path and query parameters
- If the path is `/add`, extract parameters `a` and `b`
- Return the sum as a string (e.g., "8" for 5+3)
### Function Signature
```c
void handle_request(char *request, int client_socket);
```
### Request Format
An HTTP request line looks like:
```c
GET /add?a=5&b=3 HTTP/1.1
```
The query parameters are after the `?`:
- `a=5` means parameter `a` has value `5`
- `b=3` means parameter `b` has value `3`
- Parameters are separated by `&`
### HTTP Response Format
Your response should include:
```c
HTTP/1.1 200 OK\r\n
Content-Type: text/plain\r\n
\r\n
<result as string>
```
### Notes
- Use `strstr()` to find `"a="` and `"b="` in the request
- Use `atoi()` to convert string numbers to integers
- Use `sprintf()` or `snprintf()` to convert the result back to a string
- Include `<string.h>` for string functions and `<stdlib.h>` for `atoi()`

---
## Example
|Request Path|Response Body|
|---|---|
|`GET /add?a=5&b=3 HTTP/1.1`|`8`|
|`GET /add?a=10&b=20 HTTP/1.1`|`30`|
|`GET /add?a=0&b=0 HTTP/1.1`|`0`|
|`GET /add?a=100&b=50 HTTP/1.1`|`150`|
### Test Locally
```bash
$ gcc -o calculator_server calculator_server.c http-server.c
$ ./calculator_server
Server started on port 8000

# In another terminal:
$ curl "http://localhost:8000/add?a=5&b=3"
8
$ curl "http://localhost:8000/add?a=10&b=20"
30
$ curl "http://localhost:8000/add?a=100&b=50"
150
```

---
## Code
```c
#include "http-server.h"
#include <string.h>
#include <stdlib.h>

void handle_request(char *request, int client_socket)
{
    char header[] = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n";
    if(strncmp(request, "GET /add?", 9) == 0){
        char* start = strstr(request, "?") + 1;

        char* a = strstr(start, "a=") + 2;
        char* b = strstr(start, "b=") + 2;
        int a_val = 0;
        int b_val = 0;
        sscanf(a, "%d", &a_val);
        sscanf(b, "%d", &b_val);

        int sum = a_val + b_val;
        char result[32];
        sprintf(result, "%d", sum);
        send(client_socket, header, strlen(header), 0);
        send(client_socket, result, strlen(result), 0);
    }
}
```