## Task
Implement the function `to_lowercase`
- Take a [[Computer Science/UTF-8]] encoded string and change it *in-place* so that any [[ASCII]] uppercase characters `a-z` are changed to their lowercase version
- Leave all other characters unchanged
- Return the number of characters updated from **uppercase** to **lowercase**

### Function Signature
```c
// Returns the number of characters lowercased and lowercases the capitalized 
// a-z ASCII characters of str in-place.
int32_t to_lowercase(char str[]);
```
---
## Examples
```bash
$ gcc to_lowercase.c -o to_lowercase
$ ./to_lowercase 
ABCDè
4 abcdè
éCLAIRE
6 éclaire
HELLO
5 hello
$ ./to_lowercase < small_input.txt
5 hello
3 Über
5 piÑata
0 ÜÖÅÑ
0 
1 aÜÖÅÑ
2 aÜÖÅÑb
1 ÜÖÅÑb
5 smÖrgÅs
0 😬
```
---
## Code
```c
#include <stdint.h>

int32_t to_lowercase(char str[]){
	int32_t count = 0;
	for(int i = 0; str[i] != 0; i++){
		if((str[i] & 0x80) == 0 && (str[i] >= 'A' && str[i] <= 'Z')){
			count++;
			str[i] += ('a' - 'A');
		}
	}
	
	return count;
}
```