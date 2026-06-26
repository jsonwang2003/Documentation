## Task
Implement the function `bin8_to_dec`
- Write a function that takes a null-terminated `char[]` consisting of `'0'` and `'1'` characters ($\text{length} = 8$) and returns the **signed 8-bit integer value** it represents
- The function signature
```c
// Given a char[8] representing a binary number, returns the signed 8-bit integer it represents
// For example:
//   "10000000" produces -128 (most negative 8-bit number)
//   "00000001" produces 1
//   "00000000" produces 0
//   "11111111" produces -1
//   "01111111" produces 127 (most positive 8-bit number)
int8_t bin8_to_dec(char binary[]);
```
---
## Test
```bash
$ gcc bin8_to_dec.c -o bin8_to_dec
$ ./bin8_to_dec
10000000
-128
00000001
1
11111111
-1
$ ./bin8_to_dec < small_input.txt
-128
1
0
127
-1
```
---
## Code
```c
#include <stdint.h>
#include <string.h>

int8_t bin8_to_dec(char binary[]){
	int8_t result = 0;
	int8_t value_at_index = 1;
	size_t max_len = strlen(binary);
	
	for(int i = max_len - 1; i > 0; i--, value_at_index *= 2){
		if(binary[i] == '1'){
			result += value_at_index;
		}
	}
	
	if (binary[0] == '1') {result -= 128;}
	return result;
}
```

> [!NOTE]
> Signed Integer value uses [[2's Compliment]] convention, where **the most significant bit** represents **the negative at that value**
> → In an 8-bit integer, the most significant bit is the $2^7 = 128$, therefore the **signed 8-bit integer value at the most significant bit** is $-128$

