## 1. Data Representations
While C++ and Java share many basic types, C++ exposes more hardware-level details and offers different behaviors for objects and strings.
### Numeric Types
- **Integer Size**:
    - **Java**: `int` is **always** 4 bytes.
    - **C++**: `int` size depends on the machine (though typically 4 bytes on modern systems).
- **Unsigned Types**: C++ allows `unsigned` variables, effectively shifting the range to non-negative numbers only.
	- **Signed `int`**: Range $−2^{31}$ to $+2^{31}−1$ (uses [[2's complement]]).
    - **`unsigned int`**: Range $0$ to $+2^{32}−1$ (uses all 32 bits for magnitude).

![[Pasted image 20260104221622.png]]
### Strings
C++ `string` objects are fundamentally different from Java's `String`.

|Feature|Java|C++|
|---|---|---|
|**Mutability**|Immutable (cannot change once created).|Mutable (can modify individual characters).|
|**Substring**|`s.substring(start, end)`|`s.substr(start, length)`|
|**Concatenation**|Can concatenate String + Object (auto-convert).|Can **only** concatenate string + string (no auto-convert).|

```c++
string message = "Hi, Niema!";
string name = message.substr(4, 5); // Starts at index 4, length 5 -> "Niema"
```

![[Pasted image 20260104221716.png]]
### Object Comparison
- **Java**: Uses `==` for primitives and reference equality; uses `.equals()` for object value comparison.
- **C++**: Uses `==`, `!=`, `<`, etc., for **everything**.
    - **Operator Overloading**: C++ allows classes to define custom behaviors for operators like `==`.

![[Pasted image 20260104221733.png]]

---
## 2. Variable Safety & Scope

C++ gives the programmer more freedom, which introduces new risks compared to Java's strict safety checks.
### Initialization Safety
- **Java**: Compiler throws an error if you use an uninitialized local variable.
- **C++**: **No Error.** The variable contains "garbage data" (whatever random bits were at that memory address).
### Type Safety
- **Java**: Throws an error if you "demote" a type (e.g., `double` to `int`) without a cast.
- **C++**: **No Error.** Implicit demotion occurs silently.

```c++
int harry; // Uninitialized: contains garbage data
int lloyd; // Uninitialized: contains garbage data
bool dumbAndDumber = (harry + lloyd); // VALID in C++ (but dangerous!)
```

![[Pasted image 20260106125738.png]]
### Global Variables
Unlike Java, where everything belongs to a class, C++ allows **Global Variables** declared outside any function or class.
- **Risk**: They can be accessed/modified by _any_ function, making debugging difficult.

![[Pasted image 20260106125800.png]]

---
## 3. Classes and File Structure

### Class Syntax Differences
1. **Sections**: C++ uses `public:` and `private:` labels for blocks of code, rather than labeling every individual item.
2. **Semicolon**: C++ classes must end with a semicolon `;`.
3. **Const Methods**: "Getter" methods are tagged with `const` to enforce that they do not modify instance variables.

![[Pasted image 20260106125933.png]]
### Declaration vs. Definition (.h vs .cpp)
C++ code is typically split into two files to allow code distribution without revealing the implementation.
- **Header File (`.h`)**: Contains **Declarations** (the map/interface of the class).
- **Source File (`.cpp`)**: Contains **Definitions** (the actual logic).

**Example Header (`Student.h`):**

```cpp
class Student {
    public:
        static int numStudents;    // Declaration
        Student(string n);         // Declaration
    private:
        string name;
};
```

**Example Source (`Student.cpp`):**

```cpp
// Use strict scope resolution operator ::
int Student::numStudents = 0;      // Definition

// Constructor using Member Initializer List
Student::Student(string n) : name(n) { 
    numStudents++;
}
```

> [!NOTE] Member Initializer List 
> In C++, `const` instance variables **must** be initialized using the colon (`:`) syntax before the constructor body (as seen above with `: name(n)`).
> 
> ![[Pasted image 20260106125949.png]]

---
## 4. References, Pointers, and Memory
This is the most significant difference. Java handles memory automatically via references; C++ offers three distinct ways to handle data.
### 1. Value Semantics (The Default)
In C++, variables store **values**, not references.
- **Assignment**: `Student a = b;` creates a **full copy** of object `b`.
- **Effect**: Modifying `a` does not affect `b`.

![[Pasted image 20260106220008.png]]
### 2. References (`&`)
A **Reference** is an alias (another name) for an existing variable.
- **Syntax**: `Type & refName = originalVar;`
- **Behavior**: Modifying `refName` directly modifies `originalVar`. No copying occurs.

> [!Example] Reference Example
> ```cpp
> Student s1 = Student("Niema");
> Student & s2 = s1;
> Student s3 = s2;
> ```
> 
> ![[Pasted image 20260106220154.png]]
### 3. Pointers (`*`)
A **Pointer** is a variable that stores a **memory address**.
- **Syntax**: `Type * ptr = &variable;` (use `&` to get the address).
- **Dereferencing**: Use `*ptr` to access the data at that address.
- **Arrow Operator**: Use `ptr->func()` to call methods on the object pointed to.
- **Pointer to Pointer**: You can have `int ** p`, which is a pointer to a pointer.

> [!Example]-
> ```cpp
Student lloyd("Lloyd");                // initialize Student object
Student* harry = new Student("Harry"); // initialize Student pointer
Student* ptr1 = &lloyd;                // initialize ptr1 to store the address of 'lloyd'
Student* ptr2 = harry;                 // initialize Student pointer pointing to same object as 'harry'
cout << (*ptr1).getName();             // prints "Lloyd"
cout << ptr2->getName();               // prints "Harry"
> ```
> 
> ![[Pasted image 20260104222507.png]]
### Memory Management: Stack vs. Heap
- **Automatic (Stack)**: Declared normally (`Student s;`). Deleted automatically when scope ends.
- **Dynamic (Heap)**: Declared with `new` (`Student* s = new Student();`).
    - **Manual Deletion**: You **must** call `delete s;`.
    - **Memory Leak**: If you fail to `delete`, the memory is never reclaimed.
    - **Destructors**: When `delete` is called, the class's destructor (`~Student()`) runs. This is where you should delete any internal pointers.

> [!Example] Memory Management Example
> ```cpp
> Student s1 = Student("Niema");
> Student* s2 = new Student("Ryan"); // need manually delete
> ```
> 
> ![[Pasted image 20260106220713.png]]

---
## 5. Keyword `const` Usage

### Constants (`const` vs `final`)
- **Java `final`**: Prevents reassignment of the variable reference.
- **C++ `const`**: Prevents reassignment **AND** modification of the data (for direct values).

> [!Example]
> ```cpp
> const int a = 42;
> int const b = 42;
> ```
> 
> ![[Pasted image 20260106222731.png]]
### `const` and Pointers
The placement of `const` changes its meaning, particularly with pointers.

> [!Example] `const` and Pointers Example
> ```cpp
> int a = 42;
> const int *ptr1 = &a;
> int const *ptr2 = &a;
> int* const ptr3 = &a;
> const int* const ptr4 = &a;
> ```
> 
> ![[Pasted image 20260106223709.png]]

| Syntax                  | Meaning                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| `const int * ptr`       | **Pointer is mutable**, Data is immutable. (Can point to something else, but can't change the value).         |
| `int * const ptr`       | **Pointer is immutable**, Data is mutable. (Can change the value, but must always point to the same address). |
| `const int * const ptr` | **Both** are immutable.                                                                                       |

![[Pasted image 20260106223743.png]]

### `const` and References

```cpp
int a = 42;
const int & ref1 = a;
int const & ref2 = a;
```

![[Pasted image 20260106224256.png]]

### `const` and Functions

```cpp
class Student {
	private:
		string name;
		
	public:
		Student(string n);
		string getName() const;
	
};

Student::Student(string n) : name(n) {}

string Student::getName() const {return name;}
```

Notice the line `string getName() const;` in the class declaration. This constant indicates that the function **cannot modify my object**
- cannot assign instance variable
- can only call other const functions

---
## 6. Functions
### Global Functions
C++ allows functions like `main()` to exist outside of any class.
- **Return Value**: `main()` returns an `int` that implies the **exit status**
	- $0$ implies success
	- Not $0$ implies failed
### Parameter Passing
1. **Call by Value** (Default): The function receives a **copy**. Safe, but slow for large objects.
2. **Call by Reference** (`&`): The function acts on the **original**. Fast, allows modification.
3. **Call by Const Reference** (`const &`): The function acts on the original but **cannot modify it**.
    - **Best Practice**: Use `const &` for large objects (like a genome string) to save memory without risking accidental changes.

![[Pasted image 20260106230443.png]]

---
## 7. C++ Vectors
The `vector` is C++'s version of `ArrayList`.
- **Syntax**: `vector<Type> name;`
- **Access**: Uses array-style indexing `v[0]`.
- **Safety Warning**: Unlike Java, C++ **does not check bounds**. Accessing `v[100]` in a size-10 vector will not throw an exception; it will access garbage memory and potentially crash.
- **Storage**: Elements are contiguous in memory. Assignment (`v1 = v2`) copies the **entire** vector.

```cpp
vector<int> a;
a.push_back(42); // Add to end
firstElement = a[0]; // Accessing element
a.pop_back();    // Remove from end (void return type)

vector<int> b = a; // Create new vector b and copy all elements in a
```

---
## 8. Input / Output (I/O)
C++ uses "Streams" for I/O.
- `cout`: Standard Output (console). Uses the insertion operator `<<`.
- `cin`: Standard Input (keyboard). Uses the extraction operator `>>`.
- `cerr`: Standard Error.

```cpp
int age;
cout << "Enter age: ";
cin >> age; // Reads integer from user
```

---
## 9. C++ Templates
Templates implement **Generic Programming**, allowing classes and functions to operate on any data type.
### Implementation Comparison
- **Java (Generics)**: Uses **Type Erasure**. The type is mostly a compiler-side check and is removed at runtime.
- **C++ (Templates)**: Uses **Instantiation**. The compiler uses the template as a blueprint to generate a specific class for every type used.

![[Pasted image 20260106232358.png]]
### Key Differences
- **Performance**: C++ generates specialized machine code for each type, making it faster than Java’s shared-code approach.
- **Primitives**: Templates natively support `int` and `double`, whereas Java requires wrapper classes (e.g., `Integer`).
- **Resolution**: Templates are resolved at **compile-time**. If a type doesn't support an operation used in the template, the code will not compile.