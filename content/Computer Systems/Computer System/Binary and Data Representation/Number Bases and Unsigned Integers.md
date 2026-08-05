## Decimal Numbers
Uses a *base of 10*, implying 2 important properties for the **interpretation** and **representation** of decimal values
1. Any individual **digit** in a base 10 number stores one of 10 unique values (0-9)
	- Storing a value $> 9$, the value must **carry** to an additional digit **to the left**.
2. The position of each digit in the number determines how ***important*** that digit is to the overall value of the number
	- Each successive digit (to the left) contributes a factor *ten* more than the next
	![[Pasted image 20251007212517.png]]
	1. The $5$ in the "ones" place contributes 5 ($5 \times 10^0$). 
	2. The 2 in the "tens" place contributes 20 ($2 \times 10^1$).
	3. The 4 in the "hundreds" place contributes 400 ($4 \times 10^2$).
	4. The 8 in the "thousands" place contributes 8000 ($8 \times 10^3$)
3. Which can express the value $8425$ as
	$$
	(8 \times 10^3) + (4 \times 10^2) + (2 \times 10^1) + (5 \times 10^0)
	$$
> [!NOTE]
> This pattern of **increasing exponents** applied to a base of $10$ is the reason why it's called a *base 10* number system
> Assigning position numbers to digits from **right** to **left** starting from $d_0$ → each digit $d_i$ contributes $10^i$ to the overall value

The overall value of any $N$-digit decimal number can be expressed as:
$$
(d_{N-1} \times 10^{N-1}) + (d_{N-2} \times 10^{N-2})  + ... + (d_2 \times 10^2) + (d_1 \times 10^1) + (d_0 \times 10^0)
$$
## Unsigned Binary Numbers
Uses a *base of 2* instead of decimal's 10. And we can analyze binary numbers the same way as decimal numbers
1. Any Individual bit in a *base 2* number stores up to 2 unique values (`0` or `1`)
	- To store a value larger than `1`, the binary encoding must **carry** to an additional bit to the left
	- **Counting in binary** follows the same pattern as decimal → enumerate the values and adding digits (bits)

| Binary Value | Decimal Value |
| ------------ | ------------- |
| 0            | 0             |
| 1            | 1             |
| 10           | 2             |
| 11           | 3             |
| 100          | 4             |
| 101          | 5             |
| 110          | 6             |
| 111          | 7             |
| ...          | ...           |

2. The position of each bit in the number determines how important that bit is to the numerical value of the number
	- Very similar to **decimal** → replace **10** at the base of each exponent with a **2**
$$
(d_{N-1} \times 2^{N-1}) + (d_{N-2} \times 2^{N-2}) + ... + (d_2 \times 2^2) + (d_1 \times 2^1) + (d_0 \times 2^0)
$$

## Hexadecimal
Uses a *base 16*, which is used mainly to convert *binary* representation. Also due to the base itself is a power of two ($2^4 = 16$), it's easy to map *hexadecimal* to *binary* and vice versa.
As similar to the *decimal* and *binary*, *hexadecimal* can be interpreted in a similar way
1. Any digit in a *base 16* number stores one of 16 unique values
	- Digits 0-9 represents the values 0-9
	- Characters A-F represents the values 10-15 respectively
	- To store a value larger than 15, the number must **carry** to an additional digit to the left
2. The position of each digit in the number determines **how important that digit is to the numerical value of the number**
$$
(d_{N-1} × 16^{N-1}) + (d_{N-2} × 16^{N-2}) + …​ + (d_2 × 16^2) + (d_1 × 16^1) + (d_0 × 16^0)
$$

>[!WARNING]
>###### Hexadecimal Misconception
>Memory Addresses are most common to be represented in *hexadecimal* output. But memory address are **still stored using *binary* in the hardware**, just like other data

> [!IMPORTANT]
> ###### Distinguishing Number Bases
> One potential problem is a lack of clarity for **how to interpret number**
> 
> Example:  
> Consider the value $1000$. It is not immediately obvious whether to interpret number as *decimal*, a *binary*, or a *hexadecimal*
> 
> To Indicate which numbering systems we use to represent such values, we **attach a prefix to all non-decimal numbers**:
> - **0b**: The prefix for *binary* number> - **0x**: The prefix for *hexadecimal* numbers

---
## Storage Limitations
A programmer must choose **how many bits** to dedicate to a variable **prior** to storing it
- Before storing a value, a program **must allocate storage space for it**. 
	- In C, declaring a variable tells the compiler **how much memory** it needs based on its type
- Hardware storage devices have **finite capacity**. 
	- A storage locations inside the CPU that are used as **temporary "scratch space"** ([[registers]]) are more constrained. 
	- A CPU uses registers that are **limited to its word size** (`32` or `64` bits depending on CPU architecture)
- Programs often **move data from one storage device to another**
	- As values get larger, storage devices need **more physical wires** to communicate signals between each other for memory locations
	- Expanding storage **increases complexity** of the hardware → leaves less physical space for other components
### Representing Data
Due to this limitation, the number of bits used to store an integer dictates **the range of its representable values**
![[Pasted image 20251007222146.png]]
- **Left Picture**: Represents an *infinite* unsigned number line
- **Right Picture**: Represents a *finite* unsigned number line
	- "wraps around" at either endpoint, which can cause **[[Integer Overflow|overflow]]

Since there is a limit to the amount of bits a variable can possess → what is the largest positive value that $N$ bits can store?

For any given integer value $N$, there are exactly $2^N$ unique bit patterns. However one of such unique bit pattern represents the value 0. Therefore that leaves the maximum positive integer value of $2^N -1$ → positive values ranges from $[1, 2^N-1]$