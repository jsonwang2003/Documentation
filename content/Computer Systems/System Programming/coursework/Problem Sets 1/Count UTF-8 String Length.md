## Task
Implement the function `utf8_strlen`
- Given a `char[]` representing a [[Computer Systems/System Programming/Strings and Data/UTF-8]] encoded string
- Return the **number of UTF-8 codepoints** (characters) in the string
- Input will always be a **valid UTF-8** and will not exceed `2048` bytes

### Function Signature
```c
// Given a char[] representing a UTF-8 encoded string,
// return the number of UTF-8 codepoints (characters) in the string.
// The input will always be valid UTF-8 and will not exceed 2048 bytes.
//
// UTF-8 encoding rules:
// - Single byte (ASCII): 0xxxxxxx (0x00-0x7F)
// - Two bytes: 110xxxxx 10xxxxxx (0xC0-0xDF followed by 0x80-0xBF)
// - Three bytes: 1110xxxx 10xxxxxx 10xxxxxx (0xE0-0xEF followed by two 0x80-0xBF)
// - Four bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx (0xF0-0xF7 followed by three 0x80-0xBF)
//
// For example:
//   "Hello" -> 5 (5 ASCII characters)
//   "Café" -> 4 (3 ASCII + 1 two-byte character)
//   "🌟" -> 1 (1 four-byte character)
int32_t utf8_strlen(char str[]);
```
---
## Examples
```bash
$ gcc utf8_strlen.c -o utf8_strlen
$ ./utf8_strlen
Hello
5
Café
4
🌟
1
$ ./utf8_strlen < small_input.txt
5
4
1
$ # The next command is how you should create the output files
$ # It will result in a new file with the output from running ./utf8_strlen, which
$ # the grader will check for. You can open the files with vim to check the results!
$ ./utf8_strlen < small_input.txt > small_result.txt
$ ./utf8_strlen < input.txt > result.txt
```
---
## Code
```c
#include <stdint.h>

// Given a char[] representing a UTF-8 encoded string,
// return the number of UTF-8 codepoints (characters) in the string.
// The input will always be valid UTF-8 and will not exceed 2048 bytes.
int32_t utf8_strlen(char str[]){
	int32_t count = 0;
	for(int i = 0; str[i] != 0; i++, count++){
		char c = str[i];
		if((c & 0xE0) == 0xC0){ i += 1; }
		if((c & 0xF0) == 0xE0){ i += 2; }
		if((c & 0xF8) == 0xF0){ i += 3; }
	}
	return count;
}
```