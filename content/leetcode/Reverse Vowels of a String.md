---
tags:
  - TwoPointers
  - String
Difficulty: Easy
---
## Description
[Reverse Vowels of a String - LeetCode](https://leetcode.com/problems/reverse-vowels-of-a-string/description/?envType=study-plan-v2&envId=leetcode-75)

Given a string `s`, reverse only all the vowels in the string and return it.

The vowels are `'a'`, `'e'`, `'i'`, `'o'`, and `'u'`, and they can appear in both lower and upper cases, more than once.

## Examples
#### **Example 1:**

**Input:** s = "IceCreAm"
**Output:** "AceCreIm"
**Explanation:**
The vowels in `s` are `['I', 'e', 'e', 'A']`. On reversing the vowels, s becomes `"AceCreIm"`.

#### **Example 2:**

**Input:** s = "leetcode"
**Output:** "leotcede"

## Constraint
- $1 <= \text{s.length} <= 3 * 10^5$
- `s` consist of **printable [[ASCII]]** characters.

## Code
```cpp
class Solution {
public:
    string reverseVowels(string s) {
        string word = s;
        int front = 0;
        int back = s.length() - 1;
        const string vowels = "aeiouAEIOU";
  
        while(front < back){
            bool frontOK = vowels.find(word[front]) != string::npos;
            bool backOK = vowels.find(word[back]) != string::npos;
            if (frontOK && backOK) {
                swap(word[front], word[back]);
                front++;
                back--;
            }
            if (!frontOK) front++;
            if (!backOK) back--;
        }
  
        return word;
    }
};
```

## Approach
1. Create result `word`, `front`, `back`, and `vowels` variables
2. Loop through with 2 pointers until there the `front` is ahead of the `back`
	1. Check if both `front` and `back` are both vowels
		- Swap the `front` and `back`
		- Update `front` and `back`
	2. Check if `front` is a vowel
		- Update `front`
	3. Check if `back` is a vowel
		- Update `back`
3. Return the result `word`
