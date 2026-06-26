> [!INFO]
> A lightweight [[Data Structures vs. Abstract Data Types]] that stores **two heterogeneous values** as a single unit. Commonly used for **returning multiple values from functions** or **storing key-value pairs** in associative containers ([[Maps and Multimaps]]).

- **Fixed Size**: Always holds exactly two elements.
- **Heterogeneous Types**: `T1` and `T2` can be different.
- **Direct Access**: Members accessed via `.first` and `.second`.
- **Utility-Oriented**: Often used in maps, tuples, and algorithms.

---

## Declaration & Initialization

```cpp
#include <utility> 
std::pair<int, std::string> p1 = {1, "Apple"};
std::pair<int, std::string> p2 = std::make_pair(2, "Banana");
std::pair<int, std::string> p3;
p3.first = 3;
p3.second = "Cherry";
```

> [!IMPORTANT]
> need to include **utility**

---
## Member Functions
- 
