- **Signed**: the set of numbers includes **negative numbers**
- **Unsigned**: the set of numbers includes **ONLY positive numbers**

|        | Unsigned | Signed |
| ------ | -------- | ------ |
| `0000` | 0        | 0      |
| `0001` | 1        | 1      |
| `0010` | 2        | 2      |
| `0011` | 3        | 3      |
| `0100` | 4        | 4      |
| `0101` | 5        | 5      |
| `0110` | 6        | 6      |
| `0111` | 7        | 7      |
| `1000` | 8        | -8     |
| `1001` | 9        | -7     |
| `1010` | 10       | -6     |
| `1011` | 11       | -5     |
| `1100` | 12       | -4     |
| `1101` | 13       | -3     |
| `1110` | 14       | -2     |
| `1111` | 15       | -1     |
> [!NOTE]
> Signed numbers are using [[2's Complement]] system
> The **most significant bit** of the signed number still **retains the value** of that bit, but becomes a **negative** number instead
> `1000` → $-8$

>[!IMPORTANT]
>There is no difference in the bit patterns between **signed** and **unsigned number** → programming languages need to interpret the number as either **signed** or **unsigned** types

---
## Practice Problem
```c
#include <string.h>
#include <stdint.h>
#include <stdio.h>

// Take 32 ASCII characters that are 0 or 1
// Return a signed 32-bit integer
int32_t bin_to_dec_signed(char str[]){
	// Loop over the characters in the string from 0 to 31
	// Take the character at the next index
	// If '1', put a 1 in the least significant bit
	// If '0', leave that bit as 0
	// Then shift everything to the left by 1
	int32_t result = 0;
	for(int i = 0; i < 32; i++){
		result = result << 1;
		if(str[i] == '1'){ result = result | 1; }
		printf("Current result is: %b\n", result);
	}
	
	return result;
}

int main(){
	// read one line of typed input to get the ASCII string for this problem
	char input[33]; // enough space for 32 '0' or '1' characters followed by the '\0' bit
	fgets(input, 33, stdin);
	printf("Result: %d\n", bin_to_dec_signed(input));
	return 0;
}
```

- `fgets`: Wait until user types in input and press `ENTER` before continuing the rest of the program
	- `input`: The string variable to store the input
	- `33`: How much characters to read in
	- `stdin`: The default input stream (terminal)