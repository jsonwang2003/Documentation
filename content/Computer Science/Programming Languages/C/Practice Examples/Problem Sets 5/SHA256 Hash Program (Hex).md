## Task
Write a program in `main.c` that takes space-separated strings as command-line arguments, and for each one calculates the `SHA256` hash and prints the first **4 bytes** of the hash in **hexadecimal**, prefixed by `0x` and followed by a newline. If no strings are passed as arguments, nothing should be printed in the output.

---
## Example
```bash
$ gcc main.c -o main -lcrypto
$ ./main secret password
0x2bb80d53       # output line for secret whose hash starts with "2bb80d.."
0x5e884898       # output line for password whose hash starts with "5e8848.."
```

---
## Code
```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>

int main(int argc, char** argv){
    for(int i = 1; i < argc; i++){
	unsigned char hash[32];
	SHA256((unsigned char*)argv[i], strlen(argv[i]), hash);

	printf("0x%02x%02x%02x%02x\n", hash[0], hash[1], hash[2], hash[3]);
    }
}
```