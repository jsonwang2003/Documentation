> [!INFO] Definition 
> A data type defined solely by its **behavior** from the perspective of the _user_. It specifies what functions the data must provide (e.g., "I need to be able to add an item" or "I need to remove the last item"), without dictating how those functions are coded.

### ADT vs. Data Structure

The distinction between these two concepts is fundamental to software engineering:

- **Abstract Data Type (The "What")**: A logical model or interface that describes a set of features and operations (e.g., a [[Array Lists|Lists]], a [[Computer Science/Data Structures/Introductory Data Structures/Stack|Stack]], or a [[Queues]]).
- **Data Structure (The "How")**: The physical implementation or backbone used to realize those features (e.g., an [[Computer Science/Programming Languages/C/Syntax/Arrays|Arrays]], a [[Linked List]], or a [[Circular Arrays]]).

---
### Implementation Choice: The "Song List" Example

Consider an ADT designed to store a **List of Songs**. While the "behavior" (storing and playing songs) remains the same for the user, the programmer must choose the underlying structure based on performance needs:

| Requirement             | Preferred Data Structure                     | Reasoning                                                                                                         |
| ----------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Random Access**       | [[Array Lists#Array\|Array]]-based Structure | Allows the user to jump to a specific song in the middle of the list in O(1) time.                                |
| **Frequent Insertions** | [[Linked List]]                              | If the user constantly adds/removes songs from the beginning, a list avoids the O(n) shifting required by arrays. |

**The Trade-off:** You can use a Linked List to implement a Song List ADT; the functionality will be **correct**, but it will be **slower** for random access (O(n)) compared to an array.

---
### Summary of the ADT Workflow
1. **Define Features**: Identify what the user needs to do with the data.
2. **Select Backbone**: Choose the most efficient **Data Structure** for those specific operations.
3. **Implement**: Write the code that maps the ADT functions to the Data Structure's mechanics.