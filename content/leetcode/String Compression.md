---
tags:
  - TwoPointers
  - String
Difficulty: Medium
---
## Description
[String Compression - LeetCode](https://leetcode.com/problems/string-compression/description/?envType=study-plan-v2&envId=leetcode-75)

Given an array of characters `chars`, compress it using the following algorithm:

Begin with an empty string `s`. For each group of **consecutive repeating characters** in `chars`:

- If the group's length is `1`, append the character to `s`.
- Otherwise, append the character followed by the group's length.

The compressed string `s` **should not be returned separately**, but instead, be stored **in the input character array `chars`**. Note that group lengths that are `10` or longer will be split into multiple characters in `chars`.

After you are done **modifying the input array,** return _the new length of the array_.

You must write an algorithm that uses only constant extra space.

**Note:** The characters in the array beyond the returned length do not matter and should be ignored.

## Examples
#### **Example 1:**

**Input:** chars = ["a","a","b","b","c","c","c"]
**Output:** Return 6, and the first 6 characters of the input array should be: ["a","2","b","2","c","3"]
**Explanation:** The groups are "aa", "bb", and "ccc". This compresses to "a2b2c3".

#### **Example 2:**

**Input:** chars = ["a"]
**Output:** Return 1, and the first character of the input array should be: ["a"]
**Explanation:** The only group is "a", which remains uncompressed since it's a single character.

#### **Example 3:**

**Input:** chars = ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
**Output:** Return 4, and the first 4 characters of the input array should be: ["a","b","1","2"].
**Explanation:** The groups are "a" and "bbbbbbbbbbbb". This compresses to "ab12".

## Constraints
- $1 <= \text{chars.length} <= 2000$
- `chars[i]` is a lowercase English letter, uppercase English letter, digit, or symbol.

## Code
```cpp
class Solution {
public:
    int compress(vector<char>& chars) 
        int ans = 0;

        for (int i = 0; i < chars.size();){
            const char letter = chars[i];
            int count = 0;

            while (i < chars.size() && letter == chars[i]){
                count++;
                i++;
            }

            chars[ans++] = letter;
            if (count > 1){
                for (char c : to_string(count)){
                    chars[ans++] = c;
                }
            }
        }

        return ans;
    }
};
```

## Approach
1. Create 2 Pointers to track the results
	- `i`: goes through the entire `chars` array
	- `ans`: tracks the resulting array that is in chars
2. Create necessary variables to track the repeating letter
	- `letter`: the current letter repeating
	- `count`: how many times the current letter is repeating
3. Loop through until the letter is no longer repeating 
	- Due to change of letter or end of `chars` array
	- Update the `count` and `i`
4. Step out of loop → end of a letter's repetition
	- Replace the `chars[ans]` with the current letter
	- Increment the `ans` pointer
	- Combine into `chars[ans++] = letter;`
5. Insert how many repetition the `letter` has
	- If there is only 1 letter `count == 1`
		- no need to add count to the array
	- Else
		- Loop through each digit in count
		- Add the number of counts digit by digit
		- Update and `ans` pointer
		- Combine into `chars[ans++] = c`
6. Return result