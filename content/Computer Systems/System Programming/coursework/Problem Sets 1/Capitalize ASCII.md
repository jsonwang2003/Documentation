## Task
Implement the function`capitalize_ascii`
- Take a UTF-8 encoded string and change it *in-place* so that any ASCII lowercase characters a-z are changed to their uppercase versions. 
- Leave all other characters unchanged
- Return the number of characters updated from lowercase to uppercase
- The function signature is
```c
// Returns the number of cahracters capitalized and capitalizes the lowercase a-z ASCII characters of str in-place
int32_t capitalize_ascii(char str[]);
```
---
## Test
```bash
$ gcc capitalize_ascii.c -o capitalize_ascii
$ ./capitalize_ascii 
abcdè
4 ABCDè
éclaire
6 éCLAIRE
$ ./capitalize_ascii < small_input.txt
5 HELLO
3 üBER
5 PIñATA
0 üöåñ
0 
1 Aüöåñ
2 AüöåñB
1 üöåñB
5 SMöRGåS
0 😬
1 A
```
---
## Code
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int32_t capitalize_ascii(char str[]) {
	int changed = 0;
	int difference = 'a' - 'A';
	int i = 0;
	while(str[i] != '\0'){
		if(str[i] >= 'a' && str[i] <= 'z'){
			changed++;
			str[i] -= difference;
		}
	}
	
	return changed;
}
```