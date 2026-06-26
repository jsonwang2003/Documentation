> [!ABSTRACT]
> 
> A Code Point is the abstract numerical value of a character (e.g., U+2713). UTF-8 is the encoded representation of that value in memory. To get the code point, we must isolate the "data bits" from each byte and concatenate them using bitwise shifts.
### 1. Stripping the UTF-8 Overheads
UTF-8 bytes contain both **structural bits** (which tell us the sequence length) and **payload bits** (the actual character data).
- **Leading Byte**: The bits following the size prefix (like `110`, `1110`, or `11110`) are data.
- **Continuation Bytes**: These always start with `10`. Only the remaining 6 bits are data.

---
### 2. Implementation: The `code_point` Functions
To reconstruct the full integer, we use a bitmask to grab the payload bits and then "shift" them into their proper significance level.
#### 2-Byte Decoding (`code_point2`)
Used for characters like `ê` (`U+00AA`).
- **Byte 1**: Mask with `0x1F` (`0b00011111`) to get 5 data bits. Shift left by 6 to make room for the next byte.
- **Byte 2**: Mask with `0x3F` (`0b00111111`) to get 6 data bits.
- **Total Payload**: 11 bits.
#### 3-Byte Decoding (`code_point3`)
Used for characters like `✓` (`U+2713`).
- **Byte 1**: Mask with `0x0F` to get 4 data bits. Shift left by 12.
- **Byte 2**: Mask with `0x3F` to get 6 bits. Shift left by 6.
- **Byte 3**: Mask with `0x3F` to get 6 bits.
- **Total Payload**: 16 bits.

---
### 3. Logic Trace: The Checkmark `✓`
The checkmark is stored in memory as three bytes: `0xE2 0x9C 0x94`.
1. **Byte 1 (`0xE2`)**: `11100010`. The `1110` indicates a 3-byte sequence. Payload: `0010` (2).
2. **Byte 2 (`0x9C`)**: `10011100`. Payload: `011100` (28).
3. **Byte 3 (`0x94`)**: `10010100`. Payload: `010100` (20).

**The Math:**

$$(2 \ll 12) + (28 \ll 6) + 20 = 8192 + 1792 + 20 = \mathbf{10004}$$

In Hexadecimal, $10004$ is 0x2713.

---
### 4. Code Improvements & Corrections

```c
#include <stdio.h>
#include <stdint.h>

int32_t code_point2(char str[]){
	char c1 = str[0], c2 = str[1];
	return ((c1 & 0b00011111) << 6) + (c2 & 0b00111111);
}

int32_t code_point3(char str[]){
	char c1 = str[0], c2 = str[1], c2 = str[2];
	return ((c1 & 0x00001111) << 12) + ((c2 & 0b00111111) << 6) + (c3 & 0b00111111);
}

int32_t code_point4(char str[]){
	char c1 = str[0], c2 = str[1], c3 = str[2], c4 = str[3];
	return ((c1 & 0b00000111) << 18) + ((c2 & 0b00111111) << 12) + ((c3 & 0b00111111) << 6) + (c4 & 0b00111111);
}

int32_t codepoint_of(char str[]){
	if((str[0] & 0b10000000) == 0){ return str[0]; }
	else if((str[0] & 0b11100000) == 0b11000000) { return str[0]; }
	else if((str[0] & 0b11110000) == 0b11100000) { return code_point2(str);}
	else if((str[0] & 0b11111000) == 0b11110000) { return code_point3(str);}
	else {return code_point4(str);}
}

int main(){
	char checkmark[] = "✓"; // same as {0xE2, 0x9C, 0x94, 0x00}
	int32_t cp = codepoint_of(checkmark);
	printf("Code point: %d 0x%X\n", cp, cp);
	char e_hat[] = "ê"; // same as {0xC3, 0xAA, 0x00}
	int32_t cp2 = codepoint_of(e_hat);
	printf("Code point: %d 0x%X\n", cp2, cp2);
}
```

In your provided logic for `codepoint_of`, there is a slight indexing error in the conditional branches. The logic should use the symbol size to determine which function to call:

|**If Header is...**|**Symbol Size**|**Logic to Call**|
|---|---|---|
|`0xxxxxxx`|1 Byte|`return str[0]`|
|`110xxxxx`|2 Bytes|`return code_point2(str)`|
|`1110xxxx`|3 Bytes|`return code_point3(str)`|
|`11110xxx`|4 Bytes|`return code_point4(str)`|

> [!NOTE]
> 
> In your snippet, code_point3 has a typo: char c2 = str[1], c2 = str[2];. The second variable should be c3. Additionally, the bitwise masks in codepoint_of were shifted (e.g., calling code_point2 for a 3-byte mask).

---
### 5. Summary Table

|**Character**|**UTF-8 Bytes (Hex)**|**Code Point (Dec)**|**Code Point (Hex)**|
|---|---|---|---|
|`ê`|`C3 AA`|234|`0xEA`|
|`✓`|`E2 9C 94`|10004|`0x2713`|
|`🚀`|`F0 9F 9A 80`|128640|`0x1F680`|
