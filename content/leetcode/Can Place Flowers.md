---
tags:
  - Arrays
  - Greedy
Difficulty: Easy
---
## Description
[Can Place Flowers - LeetCode](https://leetcode.com/problems/can-place-flowers/description/?envType=study-plan-v2&envId=leetcode-75)

You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in **adjacent** plots.

Given an integer array `flowerbed` containing `0`'s and `1`'s, where `0` means empty and `1` means not empty, and an integer `n`, return `true` _if_ `n` _new flowers can be planted in the_ `flowerbed` _without violating the no-adjacent-flowers rule and_ `false` _otherwise_.

## Examples
#### **Example 1:**

**Input:** flowerbed = [1,0,0,0,1], n = 1
**Output:** true

#### **Example 2:**

**Input:** flowerbed = [1,0,0,0,1], n = 2
**Output:** false

## Constraints
- $1 <= \text{flowerbed.length} <= 2 * 10^4$
- `flowerbed[i]` is `0` or `1`.
- There are no two adjacent flowers in `flowerbed`.
- `0 <= n <= flowerbed.length`

## Code
```cpp
class Solution {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        for(int i = 0; i < flowerbed.size(); i++){
            bool left = i == 0 || flowerbed[i - 1] == 0;
            bool right = i == flowerbed.size() - 1 || flowerbed[i + 1] == 0;

            if (left && right && flowerbed[i] == 0) {
                flowerbed[i] = 1;
                n--;
            }
        }
  
        return n <= 0;
    }
};
```

## Approach
> [!NOTE]
> The description only specified the adjacent pots does not have flowers, so when considering the indexes `0` and `size() - 1`, the outer edges should automatically considered free (no flower)
1. Create a for-loop that goes through the entire array
2. Consider if the current index's `left` and `right` side are free
	- `left` is free when: 
		1. Current index is at the **left most** side of the array
		2. The left adjacent index does not have a flower (0)
	- `right` is free when:
		1. Current index is at the **right most** side of the array
		2. The right adjacent index does not have a flower (0)
3. Check if the `left`, `right`, and `current` index are all 0
	- Since already checked `left` and `right` using Boolean, just use them in the conditional
	- Replace the `current` with (1) indicating a flower can be there
	- Reduce the number of flowers still need to plant
4. Return if the number of flowers left to plant is less than or equal to 0