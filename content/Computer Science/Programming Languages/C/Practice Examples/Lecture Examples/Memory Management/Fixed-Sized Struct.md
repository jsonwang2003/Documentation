In C systems programming, a **Fixed-Sized Struct** is a structure where all data is stored directly within the struct itself on the **Stack**. This is contrasted with dynamic structs where data is stored via pointers on the **Heap**.

---
## Analysis of `ListOfDoubles` Memory

```c
#include <stdio.h>

#define MAX_LIST_SIZE 20

typedef struct ListOfDoubles {
	int size;
	double contents[MAX_LIST_SIZE];
} ListOfDoubles;

ListOfDoubles concat(ListOfDoubles d1, ListOfDoubles d2){
	ListOfDoubles result;
	result.size = d1.size + d2.size;
	
	for(int i = 0; i < d1.size; i++){
		result.contents[i] = d1.contents[i];
	}
	for(int i = 0; i < d2.size; i++){
		result.contents[i + d1.size] = d2.contents[i];
	}
	return result;
}

int main(){
	ListOfDoubles lod = {2, {3.0, 4.0}};
	ListOfDoubles lod2 = {3, {6.0, 9.0, 10.0}};
	ListOfDoubles added = concat(lod, lod2);
	for(int i = 0; i < added.size; i++){
		printf("%f ", added.contents[i]);
	}
	printf("\n");
}
```

### 1. Size Calculation
The size of a fixed-sized struct is the sum of its members' sizes (plus potential padding for alignment). In the provided example:

$$\begin{align*} \text{sizeof(ListOfDoubles)} &= \text{sizeof(int)} + (\text{MAX\_LIST\_SIZE} \cdot \text{sizeof(double)}) \\ &= 4 + (20 \cdot 8) \\ &= 4 + 160 \\ &= \mathbf{164 \text{ bytes}} \end{align*}$$

### 2. Passing and Returning by Value
When you pass a fixed-sized struct to a function or return it (as seen in your `concat` function), the **entire block of memory** is copied.
- **Function Call**: `concat(lod, lod2)` creates full copies of `lod` and `lod2` on the `concat` stack frame.
- **Return**: The `ListOfDoubles added` variable in `main` receives a **complete copy** of the `result` struct from the `concat` function.

---
## Stack Constraints and Design

### 1. The Stack Size Limit
Most Operating Systems and processes impose a strict **Stack Size Limit**, typically around a few **megabytes** ($\approx$ 1,000,000 bytes).
- If you define a fixed-sized struct that is too large (e.g., an array of 1,000,000 doubles), you risk a **Stack Overflow**.
- Because of this limit, fixed-sized structs are best suited for data with a known, **fixed maximum size** that remains relatively small.

### 2. Advantages and Use Cases

|**Feature**|**Fixed-Sized Struct**|
|---|---|
|**Allocation Speed**|Extremely fast (managed by the CPU stack pointer).|
|**Memory Locality**|High (all data is contiguous in memory).|
|**Cleanup**|Automatic (memory is reclaimed when the function returns).|
|**Best For**|Small datasets, coordinate systems (x, y, z), or fixed-length headers.|

---
## Related Files in Memory Management
- **[[Size of Struct]]**: Detailed look at how **alignment and padding** might change that 164-byte calculation to a larger multiple (like 168 or 176).
- **[[Dynamic Struct]]**: How to use `malloc` to move large data to the Heap when it exceeds stack limits.
- **[[Structs in C]]**: General syntax and initialization patterns.