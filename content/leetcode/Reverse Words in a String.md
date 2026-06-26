---
tags:
  - String
  - TwoPointers
Difficulty: Medium
---
## Description
[Reverse Words in a String - LeetCode](https://leetcode.com/problems/reverse-words-in-a-string/description/?envType=study-plan-v2&envId=leetcode-75)

Given an input string `s`, reverse the order of the **words**.

A **word** is defined as a sequence of non-space characters. The **words** in `s` will be separated by at least one space.

Return _a string of the words in reverse order concatenated by a single space._

**Note** that `s` may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.

## Examples
#### **Example 1:**

**Input:** s = "the sky is blue"
**Output:** "blue is sky the"

#### **Example 2:**

**Input:** s = "  hello world  "
**Output:** "world hello"
**Explanation:** Your reversed string should not contain leading or trailing spaces.

#### **Example 3:**

**Input:** s = "a good   example"
**Output:** "example good a"
**Explanation:** You need to reduce multiple spaces between two words to a single space in the reversed string.

## Constraints
- $1 <= \text{s.length} <= 10^4$
- `s` contains English letters (upper-case and lower-case), digits, and spaces `' '`.
- There is **at least one** word in `s`.

## Code
```cpp
class Solution {
public:
    string reverseWords(string s) {
        string result = "";
        int n = s.size();
        int front = 0;
        
        while (s[front] == ' ') {
            front++;
        }

        int back = front;
        while(back < n){
            if (s[front] == ' ') front++;
            if (s[back] == ' ' && s[front] != ' '){
                result = s.substr(front, back - front) + " " + result;
                front = back;
            }
            if (back == n - 1 && s[front] != ' '){
                result = s.substr(front, back - front + 1) + " " + result;
                front = back;
            }
            back++;
        }

        result = result.substr(0, result.size() - 1);

        return result;
    }
};
```

## Approach
1. Create necessary variables for the function
	- `result`: the resulting string
	- `n`: the size of the input string `s`
	- `front`: the front pointer pointing at the start of the word
	- `back`: the pointer pointing at the end the of word
2. Ignore the leading whitespaces by looping and checking for white spaces of at the `front` pointer
3. Initialize the `back` pointer to the front (the start of the first word)
4. Loop until the `back` pointer is out of range
	- Check if the current `front` pointer is at a white space &rarr; increment the `front` pointer
	- Check if there is a word created by the `front` and `back` pointer
		- Add the word into the `result` &rarr; "{word} {previous result}"
	- One potential scenario is when a word is at the end of the string `s` that has no trailing whitespace
		- Add the word into the result &rarr; "{word} {previous result}"
5. By the method for adding the word, there will always be a trailing white space &rarr; just simply remove it
6. Return the result