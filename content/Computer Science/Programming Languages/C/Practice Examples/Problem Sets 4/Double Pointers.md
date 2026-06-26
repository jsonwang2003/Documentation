Consider the following C code
```c
int arr[4][3] = { 
	{81, 92, 2}, 
	{30, 99, 37}, 
	{38, 19, 49}, 
	{69, 78, 40} 
};
int *pptr = &arr[2][2];
```
The base address of `arr` is `0x0000bb50`. What is the value of `pptr` in hexadecimal

Assume the `sizeof(int)` = 4

**Sol**: 
Each internal array of size 3 has a byte of 
$$
3 * sizeof(int) = 3 * 4 = 12
$$
The the difference from the starting address to `arr[2][2]` is
$$
2 * 12 + 4 + 4 = 24 + 8 = 32 \to 0x20
$$
Hence the address for `pptr` will be `0xbb70`

---
## Answer
To solve this problem, we note the base address of `arr` is `0x0000bb50`. This is the starting point of the entire 2D array in memory. To find the address of the element at `arr[2][2]`, calculate the offset as follows:  
  
1. Calculate the number of elements before the desired element:  
	- Since the array is stored in row-major order (rows are stored consecutively), calculate the number of elements in previous rows as the number of elements in a row * the number of rows we are passing, or `2 * 3`.  
	- Then, add the number of elements before the desired column within the current row: `2`.  
	- Thus, the total number of integers before the desired value is `2 * 3 + 2`.  
	- Multiply this by `4` (size of each integer) to get the byte offset: `offset = (2 * 3 + 2) * 4`.  
2. Add this offset to the base address to get the final pointer address:  
`pptr = 0x0000bb50 + offset`