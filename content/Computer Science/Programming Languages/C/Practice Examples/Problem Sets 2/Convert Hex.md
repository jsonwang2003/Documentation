## Task
Implement the function `hex_to_uint`
- The program reads hexadecimal strings from standard input (one per line, and for each one, converts it to a 32-bit unsigned integer and prints the decimal value
- Takes a string containing **8 hexadecimal characters**
- Convert it to a **32-bit unsigned integer**
- Return the **unsigned integer value**
### Function Signature
```c
// Converts an 8-character hexadecimal string to a 32-bit unsigned integer
// Example: "deadbeef" -> 3735928559
// Example: "cafebabe" -> 3405691582
unsigned int hex_to_uint(char* hex_string);
```
---
## Test
```bash
$ ./convert_hex
deadbeef
3735928559
cafebabe
3405691582
```
---
## Code
```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

unsigned int hex_to_uint(char *hex_string){
	unsigned int result = 0;
	for(int i = 0; hex_string[i] != 0; i++){
		char c = hex_string[i];
		result = result << 4;
		if(c >= '0' && c <= '9'){
			result += c - '0';
		}
		if(c >= 'a' && c <= 'f'){
			result += c - 'a' + 10;
		}
	}
	
	return result;
}

int main(int argc, char** argv){
	char buffer[100];
	
	while(1){
		char *maybe_eof = fgets(buffer, sizeof(buffer), stdin);
		if(maybe_eof == NULL){break;}
		size_t len = strlen(buffer);
		if(len > 0 && buffer[len - 1] == '\n'){
			buffer[len - 1] = 0;
		}
		unsigned int num = hex_to_uint(buffer);
		printf("%u\n", num);
	}
	
	return 0;
}
```