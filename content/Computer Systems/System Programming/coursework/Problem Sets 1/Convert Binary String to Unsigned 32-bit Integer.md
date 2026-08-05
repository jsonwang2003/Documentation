## Task
Implement the function `bin_to_dec`
- Write a function that takes a null-terminated `char[]` consisting of ASCII `'0'` and `'1'` characters ($\text{length} \leq 32$), and returns the **unsigned 32-bit integer** it represents
- If the string is less than 32 characters, assume **leading zeros up to 32 characters** (`"110"` represents `6`)
- The function signature
```c
// Given a char[] of ASCII '0' and '1' (length ≤ 32), return the unsigned 32-bit integer it represents.
// If the string is shorter than 32, assume Leading 0's
void bin_to_dec(char str[]);
```
---
## Testing
### Compiling
```bash
$ gcc bin_to_dec.c -o bin_to_dec
$ ./bin_to_dec
```
### Example Uses
```bash
$ ./bin_to_dec
110
6
00000000000000000000000000000001
1
11111111111111111111111111111111
4294967295
10000000000000000000000000000000
2147483648
```
---
## Code
```c
void bin_to_dec(char str[]) {
	int value_at_index = 1;
	int result = 0;
	size_t max_len = strlen(str);
	for (int i = max_len - 1; i >= 0; i--, value_at_index *= 2) {
		if (str[i] == '1'){
			result += value_at_index;
		}
	}
	
	printf("%u\n", result);
}
```

>[!NOTE]
> `printf("%u")` -> prints an **unsigned** integer value (%d prints **signed** integer value)

