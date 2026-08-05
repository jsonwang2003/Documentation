## Task
Implement the function `check_sha256`
- Take the string and expected hash as parameter
- Compute the SHA256 hash of the string
- Compare it to the expected hash (case-sensitive)
- Return `1` if they match, `0` if they don't
### Function Signature
```c
// Returns 1 if the SHA256 hash of input_string matches expected_hash
// Returns 0 otherwise
// Comparison should be case-insensitive
int check_sha256(char* input_string, char* expected_hash);
```

### Input Format
- Each line: `string hash`
- Example:
```bash
cse-b250 ffe45b2710a4a8b52ed81bc32fa88f7b33fa6fbfd302ca0bca5c88316f6fb3f1
hello 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
```
---
## Test
```bash
$ ./shacheck
cse-b250 ffe45b2710a4a8b52ed81bc32fa88f7b33fa6fbfd302ca0bca5c88316f6fb3f1
match
hello 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
match
```
---
## Code
```c
#include <openssl/sha.h>

int check_sha256(char *input_string, char *expected_hash)
{
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(input_string, strlen(input_string), hash);

    char result[SHA256_DIGEST_LENGTH * 2 + 1];
    for(int i = 0, j = 0; j < SHA256_DIGEST_LENGTH; i += 2, j++){
            unsigned char high = (hash[j] >> 4) & 0x0f;
            unsigned char low = hash[j] & 0x0f;

            result[i] = high < 10 ? '0' + high : 'a' + (high - 10);
            result[i + 1] = low < 10 ? '0' + low : 'a' + (low - 10);
    }
    result[SHA256_DIGEST_LENGTH * 2] = 0;

    for(int i = 0; i < SHA256_DIGEST_LENGTH * 2; i++){
        if (result[i] != expected_hash[i]){return 0;}
    }
    return 1;
}
```