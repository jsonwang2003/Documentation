## 1. 
```c
#include <stdio.h>

int main() {
	char hello[] = "Hello!"; // Strings in C is a char[] that ends in null (ASCII 0)
	char hello2[] = {72, 101, 108, 111, 33, 0}; // uses ASCII to represent the individual characters
	
	puts(hello); // puts() --> print new line function
	puts(hello2);
}
```

```bash
gcc hello.c -o hello # compiles the c program source code
./hello # Run the program
Hello!
Hello!
```

#### `gcc hello.c -o hello`
- **`gcc`**: the compiler name
- **`hello.c`**: the file in c
- **`-o hello`**: name the output file `hello` instead of `a.out`

> [!WARNING]
> `-o` is dependent of the program's integration. Generally stands for a flag for **Output**

### Note
In memory: 
- `hello1`: represented `H` in an 8 bit but specified in characters, etc.
- `hello2`: represented `H` in an 8 bit but specified in integers, etc.
## 2. 
```c
#include <stdio.h>

int main() {
	char hello[] = "Hello!";
	char hellonum[] = {72, 101, 108, 108, 111, 33, 0};
	char hellobin[] = {0b1001000, 0b1100101, 0b1101100, 0b1101100, 0b1101111, 0b100001, 0b0};
	
	puts(hello);
	puts(hellonum);
	puts(hellobin);
	
	printf("%c %c %c\n", hello[0], hello[1], hello[2]);
	printf("%d %d %d\n", hello[0], hellonum[0], hellobin[0]);
	printf("%c %c %c\n", hello[0], hellonum[0], hellobin[0]);
}
```

```bash
gcc hellobin.c -o hellobin
./hellobin

Hello!
Hello!
Hello!
H e l
72 72 72
H H H
```
# Flow of keyboard keys

Press A 
→ Closed the circuit and send a electrical signal 
→ Keyboard controller 
- Moving the letters through the buffer
→ Buffer where the last `n` keys pressed
- `n` boxes to store the last `n` characters pressed
- Each box (on a full US-keyboard layout)→ about 100 little bit boxes (0/1)
	→ ~100 bits per character (not optimal)
	→ Can we do better?
	- Use ASCII → uses **7** bits instead of **100** bits
	- With **7 bits** we can represent $2^7$ or $128$ different characters, which is **sufficient** to store most if not all characters in the US layout
> [!NOTE]
> Why not **8 bits**?
> With 8 bits, the amount of unique characters that the system can represent is 256

→ Flip-Flop Gate
- 0 and 1 signal input
- A physical gate sticks to either 0 or 1
- SRAM → Static Random Access Memory
	- **Volatile Memory**: Loses the data stored when *power off* 
	- Cheap silicon (Sand)
	- Fast (100m/times/sec)
	- Accurate (as long as the power is on)
