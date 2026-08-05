> [!ABSTRACT]
> 
> To find the recurrence relation, we model the time complexity $T(n)$ by accounting for every action the algorithm takes. We break the code down into recursive costs (calls to self) and overhead costs (everything else).

---
## The Extraction Framework

To build your equation for $T(n)$, analyze the algorithm's code and identify these three components:
### 1. Identify the Recursive Calls ($a$)
Count how many times the function calls itself in a single execution path.
- **If it calls itself once:** $T(\dots)$
- **If it calls itself twice:** $2T(\dots)$
- **Variable ($a$):** Represents the "branching factor" of your recursion tree.
### 2. Identify the Input Shrinkage ($b$ or $k$)
Look at the arguments passed to those recursive calls. How much smaller is the new input?
- **Subtractive (Linear):** If the input size decreases by a fixed amount (e.g., `n-1`), the call is $T(n-1)$.
- **Dividing (Geometric):** If the input size is halved (e.g., `n/2`), the call is $T(n/2)$.
### 3. Identify the Local Work ($f(n)$)
Calculate the cost of all operations **inside** the current function call, excluding the recursive calls. This is the "overhead" or "combine" step.
- **$O(1)$:** Constant time operations (simple comparisons, `if` statements, basic arithmetic).
- **$O(n)$:** Linear operations (a `for` loop that runs from $1$ to $n$ inside the function).

---
## Common Patterns

|**Algorithm Structure**|**Resulting Recurrence Relation**|
|---|---|
|**Linear Reduction** (e.g., `FindMax`)|$T(n) = T(n-1) + c$|
|**Double Reduction** (e.g., `Fibonacci`)|$T(n) = T(n-1) + T(n-2) + c$|
|**Divide & Conquer** (e.g., `Binary Search`)|$T(n) = T(n/2) + c$|
|**Split & Merge** (e.g., `Merge Sort`)|$T(n) = 2T(n/2) + cn$|

---
## Walkthrough: `countDoubleRec`

Let's extract the relation from your "00" counting algorithm:

Code snippet

```python
countDoubleRec(string, n):
	if n < 2:
		// Base Case
		return 0
	if string[0,1] == "00":
		// Recursive Call A
		return 1 + countDoubleRec(string[1:], n-1)
	else:
		// Recursive Call B
		return countDoubleRec(string[1:], n-1)
```

1. **Recursive Calls ($a$):** Only one path is taken (either the `if` or the `else`), so there is **1** recursive call.
2. **Input Shrinkage:** The string is sliced starting from the second character, so the size becomes **$n-1$**.
3. **Local Work:** The comparison `== "00"` and the addition `1 + ...` are constant time operations, denoted as **$c$**.

The Resulting Relation:

$$
T(n) = T(n-1) + c
$$

---
## Next Steps
Once you have extracted the relation (like $T(n) = T(n-1) + c$), you can solve for the **Closed Form** to find the Big-O:
- For **Linear** reductions ($n-1$), use **[[Unraveling]]**.
- For **Geometric** reductions ($n/b$), use the **[[Master Theorem]]**.