## Task
Implement the `main` function
- Read a string from `argv[1]` and print its **SHA256** hash in lowercase hexadecimal format

>[!IMPORTANT]
> Must compile with the `-lcrypto` flag to link the OpenSSL library
> ```c
> gcc sha256_arg.c -o sha256_arg -lcrypto
> ```

---
## Test
```bash
$ gcc sha256_arg.c -o sha256_arg -lcrypto
$ ./sha256_arg password
5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8
$ ./sha256_arg hello
2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
$ ./sha256_arg "Hello World!"
7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069
$ ./sha256_arg CSE29
8c0d6f3f5bcfed5043c0b854560dfcc3f2857a5c2d0d98d0b8f5d64c35b44aef
```
---
## Code
```c
#include <stdio.h>
#include <string.h>
#include <openssl/sha.h>

int main(int argc, char** argv) {
    unsigned char hash[32];
    SHA256(argv[1], strlen(argv[1]), hash);
    for(int i = 0; i < 32; i++){
        printf("%02x", hash[i]);
    }
    printf("\n");

    return 0;
}
```