---
tags:
  - Arrays
  - Greedy
Difficulty: Medium
---
## Description
[Increasing Triplet Subsequence - LeetCode](https://leetcode.com/problems/increasing-triplet-subsequence/description/?envType=study-plan-v2&envId=leetcode-75)

Given an integer array `nums`, return `true` _if there exists a triple of indices_ `(i, j, k)` _such that_ `i < j < k` _and_ `nums[i] < nums[j] < nums[k]`. If no such indices exists, return `false`.

## Examples
#### **Example 1:**

**Input:** nums = [1,2,3,4,5]
**Output:** true
**Explanation:** Any triplet where i < j < k is valid.

#### **Example 2:**

**Input:** nums = [5,4,3,2,1]
**Output:** false
**Explanation:** No triplet exists.

#### **Example 3:**

**Input:** nums = [2,1,5,0,4,6]
**Output:** true
**Explanation:** One of the valid triplet is (3, 4, 5), because nums[3] == 0 < nums[4] == 4 < nums[5] == 6.

## Constraints
- $1 <= \text{nums.length} <= 5 * 10^5$
- $-2^{31} <= \text{nums[i]} <= 2^{31} - 1$

## Code
```cpp
class Solution {
public:
    bool increasingTriplet(vector<int>& nums) {
        int min1 = INT_MAX;
        int min2 = INT_MAX;
        for(int n : nums){
            if (n <= min1) min1 = n;
            else if (n <= min2) min2 = n;
            else return true;
        }
        return false;
    }
};
```

## Approach
> To satisfy a triplet with such condition, there must be 2 values at minimum where the first is strictly the smallest, the second that is greater than the first but smaller than the third.
1. Create 2 variables and initialize to max (easier to kick start the minimum detection)
2. Loop through each value in the `nums` array
	1. Check if the value is less than or equal to the `min1` value (ensure there is no value that is equal to the first slips through to the second minimum)
	2. Check if the value is less than or equal to the `min2` value (again, ensure no value that is equal to the second slips though to the third)
	3. Since the previous 2 conditions have found the 2 value where both index and values are both less than the third 
	   → there is a triplet that satisfies the condition
	   → Return True
3. Return False since no such triplet is found