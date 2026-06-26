---
tags:
  - TwoPointers
  - String
Difficulty: Easy
---

## Description
[Merge Strings Alternately - LeetCode](https://leetcode.com/problems/merge-strings-alternately/description/?envType=study-plan-v2&envId=leetcode-75)

You are given two strings `word1` and `word2`. Merge the strings by adding letters in alternating order, starting with `word1`. If a string is longer than the other, append the additional letters onto the end of the merged string.

Return _the merged string._

## Examples

#### **Example 1:**

**Input:** word1 = "abc", word2 = "pqr"
**Output:** "apbqcr"
**Explanation:** The merged string will be merged as so:
word1:  a   b   c
word2:    p   q   r
merged: a p b q c r

#### **Example 2:**

**Input:** word1 = "ab", word2 = "pqrs"
**Output:** "apbqrs"
**Explanation:** Notice that as word2 is longer, "rs" is appended to the end.
word1:  a   b 
word2:    p   q   r   s
merged: a p b q   r   s

#### **Example 3:**

**Input:** word1 = "abcd", word2 = "pq"
**Output:** "apbqcd"
**Explanation:** Notice that as word1 is longer, "cd" is appended to the end.
word1:  a   b   c   d
word2:    p   q 
merged: a p b q c   d

## Constraints
- $1 <= \text{word1.length, word2.length} <= 100$
- `word1` and `word2` consist of lowercase English letters.

## Code
```cpp
class Solution {
public:
    string mergeAlternately(string word1, string word2) {
        string result = "";
        int p1 = 0;
        int p2 = 0;
        size_t s1 = word1.length();
        size_t s2 = word2.length();

        while(p1 < s1 || p2 < s2){
            if (p1 < s1){ result += word1[p1]; }
            if (p2 < s2){ result += word2[p2]; }
            p1++;
            p2++;
        }

        return result;
    }
};
```

## Approach
1. Create the `result` variable
2. Create 2 pointers `p1`, `p2` to point at each respective word's first letter
3. Create 2 sizes `s1`, `s2` to save the function call from calling too many unnecessary functions
4. Loop until neither pointer reached to the end of the word
	1. Check if the first word finished adding to the `result`
	2. Check if the second word finished adding to the `result`
	3. Update the pointers
5. Return the result