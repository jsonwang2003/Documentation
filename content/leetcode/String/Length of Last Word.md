---
tags:
  - String
  - Easy
---

## Description
[Length of Last Word - LeetCode](https://leetcode.com/problems/length-of-last-word/description/)

Given a string `s` consisting of words and spaces, return _the length of the **last** word in the string._

A **word** is a maximal substring[^1] consisting of non-space characters only.

## Examples
1. 
	- **Input:** s = "Hello World"
	- **Output:** 5
	- **Explanation:** The last word is "World" with length 5.
2. 
	- **Input:** s = "   fly me   to   the moon  "
	- **Output:** 4
	- **Explanation:** The last word is "moon" with length 4.
3. 
	- **Input:** s = "luffy is still joyboy"
	- **Output:** 6
	- **Explanation:** The last word is "joyboy" with length 6.

## Constraints
- 1 <= `s.length` <= 10^4^
- `s` consists of only English letters and spaces `' '`.
- There will be at least one word in `s`.

## Code
```cpp
class Solution {
public:
    int lengthOfLastWord(string s) {
        int end = s.length() - 1;

        while(end >= 0 && s[end] == ' '){
            end--;
        }

        int start = end;

        while(start >= 0 && s[start] != ' '){
            start--;
        }

        return end - start;
    }
};
```

[^1]: A **substring** is a contiguous **non-empty** sequence of characters within a string.
