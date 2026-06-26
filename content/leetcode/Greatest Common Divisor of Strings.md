---
tags:
  - Math
  - String
Difficulty: Easy
---

## Description
[Greatest Common Divisor of Strings - LeetCode](https://leetcode.com/problems/greatest-common-divisor-of-strings/description/?envType=study-plan-v2&envId=leetcode-75)

For two strings `s` and `t`, we say "`t` divides `s`" if and only if `s = t + t + t + ... + t + t` (i.e., `t` is concatenated with itself one or more times).

Given two strings `str1` and `str2`, return _the largest string_ `x` _such that_ `x` _divides both_ `str1` _and_ `str2`.

## Examples
#### **Example 1:**

**Input:** str1 = "ABCABC", str2 = "ABC"
**Output:** "ABC"

#### **Example 2:**

**Input:** str1 = "ABABAB", str2 = "ABAB"
**Output:** "AB"

#### **Example 3:**

**Input:** str1 = "LEET", str2 = "CODE"
**Output:** ""

## Constraints
- $1 <= \text{str1.length, str2.length} <= 1000$
- `str1` and `str2` consist of English uppercase letters.

## Code
```cpp
class Solution {
public:
    string gcdOfStrings(string str1, string str2) {
        return (str1 + str2 == str2 + str1)
            ? str1.substr(0, gcd(str1.length(), str2.length()))
            : "";
    }
};
```

## Approach
> [!NOTE]
> Utilizing an interesting observation &#8617;
> Two strings has a common divisor **If And Only If** `str1 + str2 == str2 + str1`
1. Check the two strings has a common divisor using the special rule &uarr;
	- Has a common divisor:
		- Find the greatest common denominator of the two string's size
		- Take the prefix of either strings according to the greatest common denominator
		- Return the substring
	- Does not have a common divisor:
		- Return an empty string
