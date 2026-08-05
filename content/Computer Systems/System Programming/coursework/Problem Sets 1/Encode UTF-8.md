## Task
Implement the function `encode_utf8`
- Given a number as a **32-bit unsigned integer** and a **4 byte** destination `cahr[]`, write the [[Computer Systems/System Programming/Strings and Data/UTF-8]] encoding into that array
### Function Signature
```c
// Given a number as a 32-bit unsigned integer and a 4-byte destination char[], write the UTF8 encoding into that array 
//
// UTF-8 encoding rules:
// - Single byte (ASCII): 0xxxxxxx (0x00-0x7F)
// - Two bytes: 110xxxxx 10xxxxxx (0xC0-0xDF followed by 0x80-0xBF)
// - Three bytes: 1110xxxx 10xxxxxx 10xxxxxx (0xE0-0xEF followed by two 0x80-0xBF)
// - Four bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx (0xF0-0xF7 followed by three 0x80-0xBF)
//
// For example:
// Input -> Output
// 1 -> 01 00 00 00 
// 12345 -> E3 80 B9 00 
// 1114111 -> F4 8F BF BF 
// 233 -> C3 A9 00 00
void encode_utf8(uint32_t num, char dest[]);
```
---
## Examples
```bash
$ gcc encode_utf8.c -o encode_utf8
$ ./encode_utf8
1
01 00 00 00 
12345
E3 80 B9 00 
1114111
F4 8F BF BF 
233
C3 A9 00 00
$ ./encode_utf8 < small_input.txt
F0 9F 92 A9 
41 00 00 00 
E2 98 83 00 
F4 8F BF BF 
E2 82 AC 00
```
---
## Code
```c
#include <stdint.h>

void encode_utf8(uint32_t num, char dest[]) {
    if(num < 0x80) {
        dest[0] = num;
        dest[1] = 0;
    } else if(num < 0x800) {
        dest[0] = (0xC0 | ((num & 0x7C0) >> 6));
        num = num & 0x03F;
        dest[1] = (0x80 | (num & 0x3F));
        dest[2] = 0;
    } else if(num < 0x10000) {
        dest[0] = (0xE0 | ((num & 0xF000) >> 12));
        num = num & 0x0FFF;
        dest[1] = (0x80 | ((num & 0xFC0) >> 6));
        num = num & 0x03F;
        dest[2] = (0x80 | (num & 0x3F));
        dest[3] = 0;
    } else if(num < 0x200000) {
        dest[0] = (0xF0 | ((num & 0x1C0000) >> 18));
        num = num & 0x03FFFF;
        dest[1] = (0x80 | ((num & 0x3F000) >> 12));
        num = num & 0x00FFF;
        dest[2] = (0x80 | ((num & 0xFC0) >> 6));
        num = num & 0x03F;
        dest[3] = (0x80 | (num & 0x3F));
        dest[4] = 0;
    }
}
```