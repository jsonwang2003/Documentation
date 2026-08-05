
> [!INFO]
> The **predicate, anonymous, inline** function that can be defined directly in the code **without giving it a name** → One time use

### Syntax
```cpp
[capture](parameters) -> return_type {
	// function body
}
```
- `[]` **Capture Clause**: lets the lambda access variables from the surrounding scope
	- `[=]`: Captures all external variables by **value**
	- `[&]`: Capture all external variables by **reference**
	- `[a, &b]`: Capture `a` by **value**, `b` by **reference**
	- `[]`: **No capture**, only using **local variables** 
- `(parameters)` **Parameter List**: like any function's arguments
- `-> return_type` **Return Type**: *Optional* since compiler usually infers the return type
- `{...}` **Function Body**: the logic to execute;

### Example
Using `std::for_each` function
```cpp
vector<int> v = {10, 25, 67, 26, 21, 10};
int sum = 0;
for_each(v.begin(), v.end(), [&sum](const auto& elements){sum += element;})
cout << sum << endl;
```
- `[&sum]` → captures `sum` variable by reference (since need to update it)
- `(const auto& element)` → captures each element in vector `v`
- `{sum += element;}` → Execute this code for every element in vector `v`