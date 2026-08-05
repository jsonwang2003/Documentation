> [!ABSTRACT]
> 
> Presenting an algorithm is a four-fold process: defining the Problem, detailing the Implementation, proving the Correctness, and analyzing the Complexity.

## 1. The "What" (Specification)
Before writing code, you must define the boundaries of the problem.
- **Input**: Define the data types, constraints (e.g., $n > 0$), and any preconditions (e.g., "The input list must be sorted").
- **Output**: Define exactly what the user receives back.
- **Post-condition**: Explicitly state the relationship between input and output.
    - _Example:_ "The output is a permutation of the input such that $a_i \leq a_{i+1}$."
## 2. The "How" (Description)
This is the procedural heart of your presentation.
- **Pseudocode**: Use high-level structures (loops, conditionals) without getting bogged down in language-specific syntax.
- **Abstraction**: Use descriptive variable names and comments to explain the _intent_ of each block.
- **Modularity**: If the algorithm relies on sub-routines (like `Partition` in Quicksort), describe those separately.
## 3. The "Why" (Correctness)
You must mathematically prove that the algorithm does what you claim it does.
- **Loop Invariants**: For iterative algorithms, state the property that holds true through every iteration.
    - _See:_ [[Loop Invariants]] for the 3-step proof (Base, Inductive, Termination).
- **Structural Induction**: For recursive algorithms, prove that if the algorithm works for a smaller sub-problem, it works for the current problem.
- **Bi-Directional Proof**: For **Decision Algorithms**, prove both directions:
    1. If the property exists, the algorithm returns _TRUE_.
    2. If the algorithm returns _TRUE_, the property exists.
## 4. The "When" (Complexity)
Analysis of the algorithm's consumption of resources.
- **Time Complexity**: Express the number of operations as a function of input size $n$ using **[[Asymptotic Notation]]** ($O, \Omega, \Theta$).
- **Space Complexity**: How much extra memory does the algorithm require? (e.g., In-place vs. requiring a copy).
- **Case Analysis**:
    - **Worst Case**: The absolute maximum steps (usually Big-O).
    - **Best Case**: The minimum steps (usually Big-Omega).
    - **Average Case**: Expected performance over random inputs.

---
### Summary Table: Presentation Ingredients

|**Category**|**Component**|**Goal**|
|---|---|---|
|**What**|Problem Spec|Define the contract.|
|**How**|Implementation|Show the logic (Pseudocode).|
|**Why**|Correctness|Prove the logic works (Invariants).|
|**When**|Efficiency|Predict performance (Big-O).|