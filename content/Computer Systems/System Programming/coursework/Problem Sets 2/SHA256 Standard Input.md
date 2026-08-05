## Task
Implement the `main` function
- Read strings from `stdin` (one per line) and print the **SHA256** hash of each string in lowercase hexadecimal format

> [!IMPORTANT]
> Must compile with the `-lcrypto` flag to link the OpenSSL library
> ```bash
> gcc sha256_stdin.c -o sha256_stdin -lcrypto
> ```

### Using SHA256
```c
#include <openssl/sha.h>

unsigned char hash[32];  // SHA256 produces 32 bytes
SHA256((unsigned char*)my_string, strlen(my_string), hash);

// Print as lowercase hex (2 hex digits per byte = 64 characters total)
for(int i = 0; i < 32; i++) {
    printf("%02x", hash[i]);
}
printf("\n");
```
---
## Test
```bash
$ gcc sha256_stdin.c -o sha256_stdin -lcrypto
$ ./sha256_stdin
password
5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8
hello
2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
```
---
## Code
```c
#include <stdio.h>
#include <string.h>
#include <openssl/sha.h>

int main() {
    char buffer[1000];
    while(1){
        char* maybe_eof = fgets(buffer, sizeof(buffer), stdin);
        if(maybe_eof == NULL){ break; }

        size_t len = strlen(buffer);
        if(len > 0 && buffer[len - 1] == '\n'){ buffer[len - 1] = 0; }

        unsigned char hash[32];
        SHA256((unsigned char*) buffer, strlen(buffer), hash);
        for(int i = 0; i < 32; i++){
            printf("%02x", hash[i]);
        }
        printf("\n");
    }

    return 0;
}
```