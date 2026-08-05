Representing text is straightforward using ASCII: one byte per character fits well within `char[]` and it represents most English text. However, there are many more than 256 characters in the text we use, from non-Latin alphabets (Cyrillic, Arabic, and Chinese character sets, etc.) to emojis and other symbols like €, to accented characters like é and ü.

The [UTF-8 encoding](https://en.wikipedia.org/wiki/UTF-8#Encoding) is the default encoding of text in the majority of software today. If you've opened a web page, read a text message, or sent an email [in the past 15 years](https://en.wikipedia.org/wiki/UTF-8#/media/File:Unicode_Web_growth.svg) that had any special characters, the text was probably UTF-8 encoded.

Not all software handles UTF-8 correctly! For example, Joe got a marketing email recently with a header “Take your notes further with Connectâ€‹” We're guessing that was supposed to be an ellipsis (…), [UTF-8 encoded as the three bytes 0x11100010 0x10000000 0x10100110](https://www.compart.com/en/unicode/U+2026), and likely the software used to author the email mishandled the encoding and treated it as [three extended ASCII characters](https://en.wikipedia.org/wiki/Extended_ASCII).

This can cause serious problems for real people. For example, people with accented letters in their names can run into issues with sign-in forms (check out Twitter/X account [@yournameisvalid](https://x.com/yournameisvalid?lang=en) for some examples). People with names best written in an alphabet other than Latin can have their names mangled in official documents, and need to have a "Latinized" version of their name for business in the US. Joe had trouble [writing lecture notes](https://x.com/JoePolitz/status/1841175066845069552) because LaTeX does not support UTF-8 by default.

UTF-8 bugs can and do cause security vulnerabilities in products we use every day. A simple search for UTF-8 in the CVE database of security vulnerabilities turns up [hundreds of results](https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=utf-8).
## Task
Write a program that **reads** UTF-8 input and prints out some information about it
### Sample Output
```bash
$ ./utf8analyzer
Enter a UTF-8 encoded string: My 🐩’s name is Erdős.
Valid ASCII: false
Uppercased ASCII: MY 🐩’S NAME IS ERDőS.
Length in bytes: 27
Number of code points: 21
Bytes per code point: 1 1 1 4 3 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1
Substring of the first 6 code points: My 🐩’s
Code points as decimal numbers: 77 121 32 128041 8217 115 32 110 97 109 101 32 105 115 32 69 114 100 337 115 46
Animal emojis: 🐩
```
## Code
Repository found [here](https://github.com/ucsd-cse29-fa25/pa1-utf8-jsonwang2003)
```c
#include<stdio.h>
#include<string.h>
#include<stdint.h>

int is_ascii(char str[]){
	for(int i = 0; str[i] != 0; i++){
		if ((str[i] & 0x80) == 0x80) {return 0;}
	}
	return 1;
}

void capitalize_ascii(char str[], char result[]) {
	int difference = 'a' - 'A', i;
	for(i = 0; str[i] != 0; i++){
		if(str[i] >= 'a' && str[i] <= 'z'){
			result[i] = str[i] - difference;
		} else {
			result[i] = str[i];
		}
	}
	result[i] = 0;
}

int32_t utf8_strlen(char str[]){
	int32_t count = 0;
	for(int i = 0; str[i] != 0; i++, count++){
		char c = str[i];
		if((c & 0xE0) == 0xC0){i += 1;}
		if((c & 0xF0) == 0xE0){i += 2;}
		if((c & 0xF8) == 0xF0){i += 3;}
	}
	return count;
}

uint8_t codepoint_size(char string[], int index){
	char c = string[index];
	if (c == 0){return 0;}
	if ((c & 0x80) == 0){return 1;}
	if ((c & 0xE0) == 0xC0){return 2;}
	if ((c & 0xF0) == 0xE0){return 3;}
	if ((c & 0xF8) == 0xF0){return 4;}
	return -1;
}

void utf8_substring(const char str[], char result[], int start, int end){
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
}

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

int32_t helper(const char *str, int32_t byte_index) {
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

int is_animal_emoji_at(const char str[], int index){
    int codepoint = 0;
    int i;
    for(i = 0; str[i] != 0 && codepoint < index; i++, codepoint++){
        char c = str[i];
        if ((c & 0xE0) == 0xC0) {i += 1;}
        if ((c & 0xF0) == 0xE0) {i += 2;}
        if ((c & 0xF8) == 0xF0) {i += 3;}
    }

    if((str[i] & 0xF8) == 0xF0){
        int32_t value = helper(str, i);
        if (value >= 0x1F400 && value <= 0x1F43F) {return 1;}
        if (value >= 0x1F980 && value <= 0x1F9AE) {return 1;}
    }
    return 0;
}

int main(){
	char input[2048];

	printf("Enter a UTF-8 encoded string: ");
	fgets(input, 2048, stdin);
	size_t len = strlen(input);
	if(len > 0 && input[len - 1] == '\n'){input[len - 1] = 0;}
	printf("Valid ASCII: %s\n", (is_ascii(input) == 1) ? "true" : "false");

	char upper[2048];
	capitalize_ascii(input, upper);
	printf("Uppercased ASCII: %s\n", upper);
	printf("Length in bytes: %lu\n", strlen(input));
	printf("Number of code points: %d\n", utf8_strlen(input));
	printf("Bytes per code point: ");
	for(int i = 0; input[i] != 0; i++){
		int size = codepoint_size(input, i);
		if (size > 1){
			i += size - 1;
		}
		printf("%d ", size);
	}
	char substr[2048];
	utf8_substring(input, substr, 0, 6);
	printf("\nSubstring of the first 6 code points: %s\n", substr);
	printf("Code points as decimal numbers: ");
	for(int i = 0; input[i] != 0; i++){
		int32_t codepoint_value = codepoint_at(input, i);
		if(codepoint_value != -1){
			printf("%d ", codepoint_value);
		}
	}
	printf("\nAnimal emojis: ");
	for(int i = 0; i < utf8_strlen(input); i++){
		int is_animal = is_animal_emoji_at(input, i);
		if(is_animal == 1){
			char animal_emoji[2048];
			utf8_substring(input, animal_emoji, i, i + 1);
			printf("%s ", animal_emoji);
		}
	}
	printf("\n");
}
```

---
# Design Questions
1. Another encoding of Unicode is UTF-32, which encodes all Unicode code points in 4 bytes. For things like ASCII, the leading 3 bytes are all 0's. What are some tradeoffs between UTF-32 and UTF-8?

With UTF-32, by design, **each character is 4 bytes long**. What that implies is that any character, including ASCII characters, will be 4 bytes long. This system is **easier to manipulate** as each character is 4 bytes long and therefore can just **read each character in a cluster of 4 bytes**. However when the string which the program want to read only consists of ASCII characters such as "Hello", the length of the string in bytes will be 24 (including the `'\0'` character) as suppose to 6 bytes with UTF-8. 

With UTF-8, its only the opposite of UTF-32 where it is **harder to manipulate and analyze**, but saves memory space as the space in hardware memory is finite.

2. UTF-8 has a leading `10` on all the bytes past the first for multi-byte code points. This seems wasteful – if the encoding for 3 bytes were instead `1110XXXX XXXXXXXX XXXXXXXX` (where `X` can be any bit), that would fit 20 bits, which is over a million code points worth of space, removing the need for a 4-byte encoding. What are some tradeoffs or reasons the leading `10` might be useful? Can you think of anything that could go wrong with some programs if the encoding didn't include this restriction on multi-byte code points?

If UTF-8 system does not incorporate the prefix `10` for continuation bytes, one example scenario may cause inefficiency. Consider the following function signature:
```c
// Reads in a string str and a byte_index
// Return 1 if the character at byte_index is an ascii character and 0 if it is not
int8_t is_ascii(char str[], int16_t byte_index);
```
This can work under the normal rules of UTF-8 as the programmer can check the leading bit if `str[byte_index]` is a continuation bit or not. However without the leading `10`s to indicate continuation bit, the programmer has **no real good way to check** if the current index is a character on its own or a continuation bit. Some may get clever and try to loop through the `str`'s byte index until it reaches the index and can check if the current is a continuation bit. However, this method will ultimately cause the function's efficiency to be `O(N)` instead of `O(1)` under the normal rules. 