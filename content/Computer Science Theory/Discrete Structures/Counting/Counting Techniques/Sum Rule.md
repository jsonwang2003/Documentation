> [!INFO]
> For any **disjoint** sets $A$ and $B$,  
> $[A \cup B| = |A| + |B|]$  
> Keyword: **OR**
### More Generally
To count the number of **disjointed** pairs:
- Count the number of choices in the **first disjoint set**
- Count the number of choices in the **second disjoint set**
- **Add** those two counts

> [!CAUTION]
> This rule only applies when the sets are **disjoint**—i.e., they share no elements. If sets overlap, use the **Principle of Inclusion-Exclusion** instead.

### Example
Suppose you have:
- 5 red shirts
- 7 blue shirts  
If you want to know how many shirts you can choose **either red OR blue**, the total is:
$$
[5 + 7 = 12]
$$
### Edge Case
If sets are **not disjoint**, then:
$$
[|A \cup B| = |A| + |B| - |A \cap B|]
$$
This is a basic form of the **[[Inclusion Exclusion|Inclusion-Exclusion Principle]]**.

![[Pasted image 20251001145548.png]]
