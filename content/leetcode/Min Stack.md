---
tags:
  - Stack
  - Design
Difficulty: Medium
---
## Description

[Min Stack - LeetCode](https://leetcode.com/problems/min-stack/description/)

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the `MinStack` class:

- `MinStack()` initializes the stack object.
- `void push(int val)` pushes the element `val` onto the stack.
- `void pop()` removes the element on the top of the stack.
- `int top()` gets the top element of the stack.
- `int getMin()` retrieves the minimum element in the stack.

> You must implement a solution with `O(1)` time complexity for each function.

## Examples
**Example 1:**

**Input**
```
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]
```

**Output**
`[null,null,null,null,-3,null,0,-2]`

**Explanation**
```python
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
```

## Constraints

- $-2^{31} <= \text{val} <= 2^{31} - 1$
- Methods `pop`, `top` and `getMin` operations will always be called on **non-empty** stacks.
- At most $3 * 10^4$ calls will be made to `push`, `pop`, `top`, and `getMin`

## Code
```cpp
# include <vector>
using namespace std;

class MinStack {
public:
    vector<int> stack;
    vector<int> mins;

    MinStack() {}

    void push(int val) {
        if (mins.size() == 0){
            mins.push_back(val);
        } else if (val <= mins.back()){
            mins.push_back(val);
        }
        stack.push_back(val);
    }

    void pop() {
        if (stack.back() == mins.back()) {mins.pop_back();}
        stack.pop_back();
    }

    int top() {
        return stack.back();
    }

    int getMin() {
        return mins.back();
    }
};
```

## Approach
1. Create 2 member variables for `MinStack` class
	- stack: the actual stack data structure of the class
	- mins: the stack of the minimal value in the stack
2. `push` member function
	- Check if the mins stack is empty (kick starting the mins member stack)
	- Check if the input value **is less than or equal** to the top of the mins stack
		- Push to the top of the mins stack
	- Push value to the normal stack
3. `pop` member function
	- Check if the top of the normal stack (that is about to be removed) equal to the top of the mins stack
		- Remove the top of the mins stack
	- Remove the top of the mins stack
4. `top` member function
	- Return the top of the normal stack
5. `getMin` member function
	- Return the top of the mins stack