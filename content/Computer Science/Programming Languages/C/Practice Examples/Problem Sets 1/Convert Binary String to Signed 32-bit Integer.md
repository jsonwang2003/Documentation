## Task
Implement the function `bin_to_dec_signed`
- Write a function that takes a null-terminated `char[]` consisting of ASCII `'0'` and `'1'` characters ($\text{length} \leq 32$), and return the **signed 32-bit integer** it represents (using [[2's complement]] for negatives)
- if the string is less than 32 characters, **sign-extend the most significant bit** (the leftmost bit) up to 32 characters
- The function signature is
```c
// Given a char[] of ASCII '0' and '1' (length ≤ 32), return the signed 32-bit integer it represents
// If the string is shorter than 32, assume leading 1's 
void bin_to_dec_signed(char str[]);
```
---
## Test
### Compiling
```bash
gcc bin_to_dec_signed.c -o bin_to_dec_signed
./bin_to_dec_signed
```

### Examples
```bash
$ ./bin_to_dec_signed
110
-2
00000000000000000000000000000001
1
11111111111111111111111111111111
-1
10000000000000000000000000000000
-2147483648
```
---
## Code
```c
#include <string.h>

void bin_to_dec_signed(char str[]){
	int max_len = strlen(str);
	int result = 0;
	int value_at_index = 1;
	for(int i = max_len - 1; i > 0; i--, value_at_index *= 2){
		if(str[i] == '1'){
			result += value_at_index;
		}
	}
	
	if(str[0] == '1'){
		// value_at_index has already been updated to the next digit by the loop
		result -= value_at_index;
	}
	
	printf("%d\n", result);
}
```
