---
tags:
  - Stack
  - Design
  - Queue
Difficulty: Easy
---
## Description
[Implement Stack using Queues - LeetCode](https://leetcode.com/problems/implement-stack-using-queues/description/)

Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (`push`, `top`, `pop`, and `empty`).

Implement the `MyStack` class:

- `void push(int x)` Pushes element x to the top of the stack.
- `int pop()` Removes the element on the top of the stack and returns it.
- `int top()` Returns the element on the top of the stack.
- `boolean empty()` Returns `true` if the stack is empty, `false` otherwise.

**Notes:**

- You must use **only** standard operations of a queue, which means that only `push to back`, `peek/pop from front`, `size` and `is empty` operations are valid.
- Depending on your language, the queue may not be supported natively. You may simulate a queue using a list or deque (double-ended queue) as long as you use only a queue's standard operations.

## Examples
**Example 1:**

**Input**
```
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
```
**Output**
```
[null, null, null, 2, 2, false]
```

**Explanation**
```
MyStack myStack = new MyStack();
myStack.push(1);
myStack.push(2);
myStack.top(); // return 2
myStack.pop(); // return 2
myStack.empty(); // return False
```

## Constraints
- $1 <= \text{x} <= 9$
- At most `100` calls will be made to `push`, `pop`, `top`, and `empty`.
- All the calls to `pop` and `top` are valid.

## Code
```cpp
# include <queue>
# include <vector>
using namespace std;

class MyStack {
public:
    queue<int> q;
    
    MyStack() {}

    void push(int x) {
        if (q.empty()){
            q.push(x);
        } else {
            vector<int> array;
            while (!q.empty()){ array.push_back(q.front()); q.pop(); }
            q.push(x);
            for (int i = 0; i < array.size(); i++) { q.push(array[i]); }
        }
    }

    int pop() {
        int result = q.front();
        q.pop();
        return result;
    }

    int top() {
        return q.front();
    }

    bool empty() {
        return q.empty();
    }
};
```

## Approach
1. Create a queue as per the description
2. `push` member function
	- Check if the queue is empty
		- Push to Queue
	- Otherwise
		- Create a local array for temporary storage
		- Populate the array based on the front of the queue and maintain the order
		- Push the front(_top_) into the queue
		- Restore the values from the array into the queue
3. `pop` member function
	- Create a temporary variable storing the _top_ of the queue
	- Pop the front(_top_) of the queue
	- Return the temporary variable
4. `top` member function
	- Return the front(_top_) of the queue
5. `empty` member function
	- Return whether the queue is empty by calling the queue STL function `empty()`