>[!INFO]
>In general, if I want to put `n` **indistinguishable** objects in `k` different baskets, I could instead count the number of ways of ordering
>	$k-1$ bars and $n$ stars
>Or equivalently
>	$k-1$ ones and $n$ zeros
>	
>Which we know how to count
>$$
>\binom{n + k - 1}{k - 1} = \binom{n + k - 1}{n}
>$$



Suppose we are playing a game where we can place 5 **different** knights into 3 **different** castles. How many ways can I use the knights to guard the castles?
$$
3^5
$$
Suppose now the 5 knights are now **indistinguishable** into 3 **different** castles
$$
\binom{7}{2}
$$
Configuration ⇆ fixed density binary strings
`n` knights ⇆ `n` zeros
`k` castles ⇆ `k-1` ones
##### Proof:
1. Put `0` under the 5 knights and `1` between the castles
2. Now we have a binary string that can be arranged 
3. Now we need to choose 2 `1` to place in this 7 length string
Hence the 
$$
\binom{7}{2}
$$
### Examples
If I had 5 different castles and 11 indistinguishable knights, which binary strings would this configuration map to?
$$
	\binom{11+5-1}{5-1} = \binom{15}{4}
$$

How many different configurations are there with 5 different castles and 11 indistinguishable knights if each castle gets **at least 1 knight**?
$$
	\binom{6+5-1}{5-1}
$$
Explanation:
- Start by **assigning 5 knights each to one castle** (have them go inside and make themselves comfortable). Then the remaining 6 can be **arranged as if the castles are empty**
- In other words, treat each knight castle pairing as 1 castle (k)
---
## Integer Equations
>[!INFO]
>In general, consider the equation
>$$
>a_1 + a_2 + ... + a_k = n
>$$
>where $n$ and $a_i$'s are non-negative integers
>- number of stars = RHS constant = $n$
>- number of bars  = (number of variables) - 1 = $k - 1$
>
>There are $\binom{n+k-1}{k-1}$ solutions

### Example 1
Consider the equation
$$
	a + b + c + d = 10
$$
where $a,b,c,d$ are **non-negative integers**. How many solutions does this equation have?

One can approach this problem by considering $a, b, c, d$ **each as a basket / castle** and use the [[Stars and Bars]] method

- `n` = 10
- `k` = 4
- `result` = $\binom{10 + 4 - 1}{4 - 1} = \binom{13}{3}$ 

### Example 2
Consider the equation
$$
a_1 + a_2 + ... + a_k = n
$$
where $n$ and $a_i$'s are **positive integers**

This is the same as placing $n$ soldiers into $k$ castles such that each castle must have **at least one** soldier. We can think of the number `0` already being every $a_i$ variable
### Practice Problems
How many non-negative integer solutions are there to the equation?
$$
	a_1 + a_2 + a_3 + a_4 = 18
$$
1. No restrictions $0 \leq a_i \leq 18$
	$$
	\binom{18 + 4 - 1}{4 - 1} = \binom{21}{3}
	$$
2. Each variable is positive: $0 \leq a_i \leq 18$
	$$
	\binom{18 + 4 - 4 -1}{4-1} = \binom{17}{3}
	$$
3. $a_1$ is greater than 5: $6 \leq a_1 \leq 18$, $0 \leq a_i \leq 18$ (for all $2 \leq i \leq 4$)
	$$
	\binom{15}{3}
	$$
	Explanation:
		Since $a_1$ is some value greater than or equal to 6, then that means **that n has been reduced by 6 as it is already guaranteed to be included within $a_1$**
		But since $a_1$ is not guaranteed to be 6, **$a_1$ is still within the consideration of the 4 variables**
4. $a_1$ is less than or equal to 5: $0 \leq a_i \leq 5$, $0 \leq a_i \leq 18$ (for all $2 \leq i \leq 4$)
	$$
	\binom{21}{3} - \binom{15}{3}
	$$
	Explanation:
		Notice that this problem is **a bijection** with the previous problem. Therefore, we can take all possible outcomes of the problem - the previous answer to get the same result