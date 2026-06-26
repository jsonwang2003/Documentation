---
tit: Function Templates
---

> [!INFO]
> Allows creation of a **single function** that can operate with **different data types**. This is achieved by defining a template that **specifies the function's behavior** without **committing to a specific data types**

## Syntax
`template <template-parameters> function-declaration`
- **Template Parameters**: A series of parameters separated by **commas** that can be *generic* template types by specifying either the `class` or `typename` keyword followed by an *identifier*.

> [!NOTE]
> `class` is the old way to use function templates.
> Recommend using `typename` instead ✓

- **Identifier**: A "variable class" that can be used in the generic function declaration as if it was a regular type
```cpp
template <typename T>
T sum(T a, T b){
	return a + b;
}
```
- Declaring `T` (a generic type within the template parameters enclosed in <>) allows `T` to be used anywhere in the function definition → just as any other type
	- Represents a generic type that will be **determined on the moment the template is instantiated**
- **Instantiating a template** is applying the template to create a function using particular types or values for its template parameters
	- Same syntax as calling a regular function but specifying the **template arguments** enclosed in <>
	- `name <template-arguments(function-arguments)`
```cpp
int x = sum<int>(10, 20);
```

---
# Class Templates
## Generic Class
- all instance of the `typename` in class definition will be **replaced by the type assigned by parameter**
- Once template class is defined, we can declare objects of the class by using the **same** syntax used in STL classes → **enclosing types with <>**

## Template and Inheritance
- Can be safely inherited
- When `base class == template`, derived class inherits the same `typename`

## Note
- Each function definition is **preceded** by `template <typename T>`
	- *Each function definition is itself a template*
- Class qualifier[^1] is now `ClassName<T>` instead of simply `ClassName`

## Compiler Complication
- Most compilers **do not** support templates
	- will not make connections between 
		- function declaration ⇆ function definition
		- class interface ⇆ class implementation
- Possible Solutions
	1. for **non-member functions**, use `template<typename T>`
		- before **function declaration**
		- before **function definition**
	2. for class, check **compiler specific requirements**
		- Some need **specific options set**
		- Some require **special order of arrangement of template definitions** or **other file items**
- Class templates as parameters
	- Passing object of template class as parameter
		- need to **include type in between <>**
```cpp
void print(const std::className<type>& param);
```

---
# Common Errors and Strategies
## Common Errors
1. Using mixed types of parameter with same identifiers
```cpp
swapValues(const T& param1, const T& param2); // ✓
swapValues(int param1, double param2); // ✗
```
2. Forgetting to include **.cpp file** when template class is used
```cpp
# include "Template.h"
# include "Template.cpp" // need to include this
```
3. Forgetting **member functions definitions are themselves template** and must be preceded by `template <typename T>`
```cpp
template <typename T>
class<T>::function(const T& param){...}
```

## Strategies
1. Implement functions or classes **normally** with **any datatype**
2. **Debug** functions or classes
3. Change to a template by replacing `types` with `typename`
	- **Only** replace the types that need to be replaced
	- Add `<T>` next to class qualifier[^1]


[^1]: Class Qualifier: Keywords or Syntax that modify **how class members are accessed, stored, or behave**. Often used to describe modifiers that affect **class scope, member access, and mutability**
