## Task
Implement the function `is_animal_emoji_at`
- Write a function that takes a [[Computer Science/UTF-8]] encoded null-terminated `char[]` string and an integer `index` (**codepoint index**)
- Return `1` if the codepoint at that index is an animal emoji
- Return `0` otherwise
- **Animal Emoji** are defined as codepoints in either of these 2 ranges
	-  From 🐀 (U+1F400) to 🐿️ (U+1F43F)
	- From 🦀 (U+1F980) to 🦮 (U+1F9AE)
### Function Signature
```c
// Given a UTF-8 encoded string and a codepoint index, returns 1 if the codepoint at that index is an animal emoji, 0 otherwise
int is_animal_emoji_at(const char str[], int index);
```
---
## Examples
```bash
$ ./is_animal_emoji_at
🐶🐱🐭 0
1
🐶🐱🐭 1
1
🐶🐱🐭 2
1
🐶🐱🐭 3
0
🦀🦁🦒🦮 0
1
🦀🦁🦒🦮 3
1
🦀🦁🦒🦮 4
0
cat 0
0
a 0
0
🐶a🐱b 1
0
🐶a🐱b 2
1
🐶a🐱b 3
0
hello 0
0
hello 4
0
🦯 0
0
🐿️ 1
0
```
---
## Code
```c
int32_t codepoint_at(const char *str, int32_t byte_index) {
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

// Implement the is_animal_emoji_at function here!
// Your function should match the following signature:
// It should return 1 if the codepoint at the given index is an animal emoji, 0 otherwise.
// You may add any helper functions you need.
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
        int32_t value = codepoint_at(str, i);
        if (value >= 0x1F400 && value <= 0x1F43F) {return 1;}
        if (value >= 0x1F980 && value <= 0x1F9AE) {return 1;}
    }
    return 0;
}
```