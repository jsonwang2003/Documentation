## Task
Implement the function `hex_to_bytes`
- Take a string containing **8 hexadecimal characters**
- Convert it to **4 bytes** (each pair of hex characters by spaces)
- Print each byte as a decimal number separated by spaces
### Function Signature
```c
// Converts an 8-character hexadecimal string to 4 bytes
// Prints each byte as a decimal number separated by spaces
// Example: "deadbeef" prints "222 173 190 239"
// Example: "cafebabe" prints "202 254 186 190"
void hex_to_bytes(char* hex_string);
```
---
## Test
```bash
$ ./convert_hex_to_byte
deadbeef
222 173 190 239
cafebabe
202 254 186 190
```
---
## Code
```c
void hex_to_bytes(char *hex_string)
{
    int nums[4];
    int num = 0, count = 0;
    for(int i = 0; hex_string[i] != 0; i++){
        char c = hex_string[i];
        if(i % 2 == 0){
            if(c >= '0' && c <= '9'){
                num += c - '0';
            }
            if(c >= 'a' && c <= 'f'){
                num += c - 'a' + 10;
            }
            num = num << 4;
        } else {
            if(c >= '0' && c <= '9'){
                num += c - '0';
            }
            if(c >= 'a' && c <= 'f'){
                num += c - 'a' + 10;
            }
            nums[count] = num;
            num = 0;
            count++;
        }
    }
    for(int i = 0; i < 4; i++){
        printf("%d ", nums[i]);
    }
    printf("\n");
}
```