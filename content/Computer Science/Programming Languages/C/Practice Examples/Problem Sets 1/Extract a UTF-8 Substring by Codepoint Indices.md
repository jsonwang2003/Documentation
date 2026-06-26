## Task
- Given a **UTF-8** encoded null-terminated `char[]` string, and two integers `start` (inclusive) and `end` (exclusive) representing codepoint indices
- Print the substring between `start` and `end` to standard output (`stdout`), follow by a newline
- Substring should be null-terminated in a local buffer **that you declare inside your function**
### Function Signature
```c
// Given a UTF-8 encoded string, extract the substring from codepoint index start (inclusive) to end (exclusive)
// and print it to standard output (stdout), followed by a newline. The substring should be null-terminated in a local buffer.
void utf8_substring(const char str[], int start, int end);
```
---
## Examples
```bash
$ ./utf8_substring
café 1 3
af
你好世界 1 3
好世
Привет 2 5
иве
```
---
## Code
```c
#include <string.h>
#include <stdio.h>

// Extracts the substring from codepoint index start (inclusive) to end (exclusive)
void utf8_substring(const char str[], int start, int end) {
        char result[strlen(str)];
        int result_index = 0;
        int codepoint = 0;
        for(int byte = 0; str[byte] != 0 && codepoint < end; byte++){
                char c = str[byte];
                if ((c & 0x80) == 0) {
                        if(codepoint >= start) {
                                result[result_index] = str[byte];
                                result_index++;
                        }
                        codepoint++;
                } else if ((c & 0xE0) == 0xC0) {
                        if (codepoint >= start){
                                for(int i = 0; i < 2; i++, result_index++, byte++) {
                                        result[result_index] = str[byte];
                                }
                                byte--;
                        }
                        codepoint++;
                } else if ((c & 0xF0) == 0xE0) {
                        if(codepoint >= start){
                                for (int i = 0; i < 3; i++, result_index++, byte++) {
                                        result[result_index] = str[byte];
                                }
                                byte--;
                        }
                        codepoint++;
                } else if ((c & 0xF8) == 0xF0) {
                        if(codepoint >= start) {
                                for(int i = 0; i < 4; i++, result_index++, byte++) {
                                        result[result_index] = str[byte];
                                }
                                byte--;
                        }
                        codepoint++;
                }
        }
        result[result_index] = 0;
        printf("%s \n", result);
}
```