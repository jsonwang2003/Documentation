---
tags:
  - Greedy
---
> [!INFO]
> A **Greedy Algorithm** builds a solution step-by-step by choosing the locally optimal option at each stage, aiming for a global optimum. It’s efficient and often surprisingly effective, especially in problems with **greedy-choice property** and **optimal substructure**.

### [[Computer Science/Algorithms/Greedy|Greedy Algorithms]]

## Properties

- **Locally Optimal Decisions**: Chooses the best option at each step without reconsidering previous choices
- **No Backtracking**: Once a choice is made, it’s never revisited
- **Optimal Substructure**: The optimal solution to the problem contains optimal solutions to subproblems
- **Greedy-Choice Property**: A global optimum can be arrived at by choosing local optima
- **Fast and Simple**: Often faster than dynamic programming or exhaustive search
- **Not Always Correct**: Only works when problem structure guarantees correctness

## Common Applications

- **Activity Selection**: Choose the maximum number of non-overlapping intervals
- **Huffman Coding**: Build optimal prefix codes for data compression
- **Minimum Spanning Tree**: Kruskal’s and Prim’s algorithms
- **Dijkstra’s Algorithm**: Shortest path in weighted graphs (with non-negative weights)
- **Fractional Knapsack**: Maximize value with divisible items
- **Job Scheduling**: Maximize profit or minimize lateness

## Common Operations

- **Sort by Heuristic**: Order elements by weight, value, deadline, etc.
- **Iterate & Select**: Traverse sorted list, pick if valid
- **Accumulate**: Build partial solution (e.g., total weight, cost, path)
- **Terminate Early**: Stop once constraints are violated or goal is reached