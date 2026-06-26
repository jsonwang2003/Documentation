---
tags:
  - Arrays
  - PrefixSum
Difficulty: Medium
---
## Description
[Product of Array Except Self - LeetCode](https://leetcode.com/problems/product-of-array-except-self/description/?envType=study-plan-v2&envId=leetcode-75)

Given an integer array `nums`, return _an array_ `answer` _such that_ `answer[i]` _is equal to the product of all the elements of_ `nums` _except_ `nums[i]`.

The product of any prefix or suffix of `nums` is **guaranteed** to fit in a **32-bit** integer.

You must write an algorithm that runs in `O(n)` time and without using the division operation.

## Examples
#### **Example 1:**

**Input:** nums = [1,2,3,4]
**Output:** [24,12,8,6]

#### **Example 2:**

**Input:** nums = [-1,1,0,-3,3]
**Output:** [0,0,9,0,0]

## Constraints
- `2 <= nums.length <= 105`
- `-30 <= nums[i] <= 30`
- The input is generated such that `answer[i]` is **guaranteed** to fit in a **32-bit** integer.

## Code
```cpp
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> ans(nums.size(), 1);
        int prefix = 1;
        for (int i = 0; i < nums.size(); i++){
            ans[i] *= prefix;
            prefix *= nums[i];
        }

        int suffix = 1;
        for (int i = nums.size() - 1; i >= 0; i--){
            ans[i] *= suffix;
            suffix *= nums[i];
        }

        return ans;
    }
};
```

## Approach
1. Create the `ans` array to store the result
	- Initialize the size to match the size of `nums` and have all values start from $1$
2. Loop through the `nums` array forward
	- Update the `ans[i]` before changing the prefix (skipping the `nums[i]`)
	- The product of the values before `nums[i]` is multiplied by prefix
3. Loop through the `nums` array backward
	- Update the `ans[i]` before changing the suffix (skipping the `nums[i]`)
	- The product of the values before `nums[i]` is multiplied by suffix
4. Return the `ans`