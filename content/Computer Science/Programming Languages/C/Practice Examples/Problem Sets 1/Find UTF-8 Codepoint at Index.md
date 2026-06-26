## Task
Implement `codepoint_at` function
- Given a `char[]` representing a **UTF-8 encoded string**, and a **byte index**
- Return the **decimal codepoint value** of the Unicode character that **starts at that byte index**, or -1 if the byte at that index **cannot be the start of a valid UTF-8 character**.
- The input will always be valid UTF-8 and will not exceed 2048 bytes.
- The byte index will always be valid (less than the string's byte length).

### UTF-8 encoding **rules**
- **Single Byte (ASCII)**: `0xxxxxxx` (`0x00`-`0x7F`)
- **Two Bytes**: `110xxxxx 10xxxxxx` (`0xC0`-`0xDF` followed by `0x80`-`0xBF`)
- **Three Bytes**: `1110xxxx 10xxxxxx 10xxxxxx` (`0xE0`-`0xEF` followed by **two** `0x80`-`0xBF`)
- **Four bytes**: `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` (`0xF0`-`0xF7` followed by **three** `0x80`-`0xBF`)

### Function Signature
```c
// Given a char[] representing a UTF-8 encoded string, and a byte index,
// return the decimal codepoint value of the Unicode character that starts at that byte index,
// or -1 if the byte at that index cannot be the start of a valid UTF-8 character.
// The input will always be valid UTF-8 and will not exceed 2048 bytes.
// The byte index will always be valid (less than the string's byte length).

// UTF-8 encoding rules:
// - Single byte (ASCII): 0xxxxxxx (0x00-0x7F)
// - Two bytes: 110xxxxx 10xxxxxx (0xC0-0xDF followed by 0x80-0xBF)
// - Three bytes: 1110xxxx 10xxxxxx 10xxxxxx (0xE0-0xEF followed by two 0x80-0xBF)
// - Four bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx (0xF0-0xF7 followed by three 0x80-0xBF)

// For example:
//   "Hello", 0 -> 72 (H has codepoint value 72)
//   "Hello", 4 -> 111 (o has codepoint value 111)
//   "Café", 3 -> 233 (é has codepoint value 233)
//   "🌟star", 0 -> 127775 (🌟 has codepoint value 127775)
//   "🌟star", 1 -> -1 (byte is 10xxxxxx continuation byte)
//   "🌟star", 4 -> 115 (s has codepoint value 115)
//   "é🌟a", 2 -> 127775 (🌟 has codepoint value 127775)
//   "é🌟a", 3 -> -1 (byte is 10xxxxxx continuation byte)
int32_t codepoint_at(char str[], int32_t byte_index);
```
---
## Examples
```bash
$ gcc codepoint_at.c -o codepoint_at
$ ./codepoint_at
Hello 0
72
Hello 4
111
🌟star 0
127775
🌟star 1
-1
🌟star 4
115
é🌟a 2
127775
é🌟a 3
-1
café 3
233
$ ./codepoint_at < small_input.txt
72
111
127775
-1
233
$ # The next command is how you should create the output files
$ # It will result in a new file with the output from running ./codepoint_at, which
$ # the grader will check for. You can open the files with vim to check the results!
$ ./codepoint_at < small_input.txt > small_result.txt
$ ./codepoint_at < input.txt > result.txt
```
---
## Code
```c
#include <stdint.h>

int32_t codepoint_at(char str[], int32_t byte_index) {
	int32_t c = (unsigned char)str[byte_index];
	if(c < 0x80){return c;}
	if(c >= 0xC0 && c < 0xE0){
		return (c & 0x1F) << 6
			| ((unsigned char)str[byte_index + 1] & 0x3F);
	}
	if(c >= 0xE0 && c < 0xF0){
		return (c & 0x0F) << 12
			| ((unsigned char)str[byte_index + 1] & 0x3F) << 6
			| ((unsigned char)str[byte_index + 2] & 0x3F);
	}
	if(c >= 0xF0 && c < 0xF8){
		return (c & 0x07) << 18
			| ((unsigned char)str[byte_index + 1] & 0x3F) << 12
			| ((unsigned char)str[byte_index + 2] & 0x3F) << 6
			| ((unsigned char)str[byte_index + 3] & 0x3F);
	}
	return -1;
}
```