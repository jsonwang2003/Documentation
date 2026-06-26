---
tags:
  - String
Difficulty: Easy
---

## Description
[Length of Last Word - LeetCode](https://leetcode.com/problems/length-of-last-word/description/)

Given a string `s` consisting of words and spaces, return _the length of the **last** word in the string._

A **word** is a maximal substring[^1] consisting of non-space characters only.

## Examples
#### **Example 1:**

**Input:** s = "Hello World"
**Output:** 5
**Explanation:** The last word is "World" with length 5.

#### **Example 2:**

**Input:** s = "   fly me   to   the moon  "
**Output:** 4
**Explanation:** The last word is "moon" with length 4.

#### **Example 3:**

**Input:** s = "luffy is still joyboy"
**Output:** 6
**Explanation:** The last word is "joyboy" with length 6.

## Constraints
- $1 <= \text{s.length} <= 10^4$
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

## Approach
> [!IMPORTANT]
> Need to watchout for **Trailing Whitespaces**

1. Create the `end` pointer to point at the last character in the string
2. Traverse the pointer **backward** until the `end` pointer is pointing to a non-whitespace
3. Create the `start` pointer to point at the last character of the last word (where `end` pointer is pointing at)
4. Traverse the `start` pointer until pointing at a whitespace or start of the string
5. return the `end` index - `start` index

[^1]: A **substring** is a contiguous **non-empty** sequence of characters within a string.