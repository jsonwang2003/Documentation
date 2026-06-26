## Task
Write a program in `main.c` that takes space-separated strings as command-line arguments, and for each one calculates the `SHA256` hash and prints the first **3 bytes** of the hash as **space-separated unsigned decimal** numbers, followed by a newline. If no strings are passed as arguments, nothing should be printed in the output.

---
## Example
```bash
$ gcc main.c -o main -lcrypto
$ ./main seCret password
162 195 176      # output line for seCret whose hash starts with "a2c3b0.."
94 136 72        # output line for password whose hash starts with "5e8848.."
```

---
## Code
```c
#include <stdio.h>
#include <string.h>
#include <openssl/sha.h>
#include <stdint.h>

int main(int argc, char** argv){
    for(int i = 1; i < argc; i++){
        unsigned char hash[32];
        SHA256((unsigned char*)argv[i], strlen(argv[i]), hash);

        printf("%d %d %d\n", hash[0], hash[1], hash[2]);
    }
}
```