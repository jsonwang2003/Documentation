> [!ABSTRACT]
> 
> A **Ternary Search Tree** is a hybrid data structure that combines the prefix-searching logic of a [[Multiway Trie]] with the space-efficient storage of a [[Binary Search Tree (BSTs)]]. Each node stores a single character and has exactly three potential children: **Left**, **Middle**, and **Right**.

---
## 1. Structural Logic
In a TST, the relationship between a node and its children is determined by character comparisons and word progression:
- **Left Child:** Stores characters that are **alphabetically smaller** than the current node's character.
- **Right Child:** Stores characters that are **alphabetically larger** than the current node's character.
- **Middle Child:** Represents the **next character** in the current word string.
- **Word Nodes:** Nodes representing the end of a valid word are marked (e.g., colored blue).

![[Pasted image 20260126124053.png]]

---
## 2. Core Operations

### Find
To search for a key, we compare the current character of the key with the current node's label:
1. **If Key Char < Node Label:** Move to the **Left** child.
2. **If Key Char > Node Label:** Move to the **Right** child.
3. **If Key Char == Node Label:** * If this is the last character of the key, check if the node is a **word-node**.
    - Otherwise, move to the **Middle** child and move to the next character in the key.

```cpp
find(key): // return True if key exists in this TST, otherwise return False
    node = root node of the TST
    letter = first letter of key
    loop infinitely:
        // left child
        if letter < node.label:
            if node has a left child:
                node = node.leftChild
            else:
                return False     // key cannot exist in this TST

        // right child
        else if letter > node.label:
            if node has a right child:
                node = node.rightChild
            else:
                return False     // key cannot exist in this TST

        // middle child
        else:
            if letter is the last letter of key and node is a word-node:
                return True      // we found key in this TST!
            else:
                if node has a middle child and letter is not the last letter of key:
                    node = node.middleChild
                    letter = next letter of key
                else:
                    return False // key cannot exist in this TST
```

> [!Example]- Walk Through Example (Success)
> Below is the same example from before, and we will step through the process of finding the word _mid_:
> 
> ![[Pasted image 20260126124507.png]]
> 
> 1. We start with _node_ as the root node ('c') and _letter_ as the first letter of _mid_ ('m')
> 2. _letter_ ('m') is greater than the label of _node_ ('c'), so set _node_ to the right child of _node_ ('m')  
> 3. _letter_ ('m') is equal to the label of _node_ ('m'), so set _node_ to the middle child of _node_ ('e') and set _letter_ to the next letter of _mid_ ('i')  
> 4. _letter_ ('i') is greater than the label of _node_ ('e'), so set _node_ to the right child of _node_ ('i')  
> 5. _letter_ ('i') is equal to the label of _node_ ('i'), so set _node_ to the middle child of _node_ ('n') and set _letter_ to the next letter of _mid_ ('d')  
> 6. _letter_ ('d') is less than the label of _node_ ('n'), so set _node_ to the left child of _node_ ('d') 
> 7. _letter_ ('d') is equal to the label of _node_ ('d'), _letter_ is already on the last letter of _mid_ ('d'), and _node_ is a "word node," so **success**!
> 

> [!Example]- Walk Through Example (Fail)
> Using the same example as before, let's try finding the word _cme_, which might _seem_ like it exists, but it actually doesn't:
> 
> ![[Pasted image 20260126124559.png]]
> 
> 1. We start with _node_ as the root node ('c') and _letter_ as the first letter of _cme_ ('c')
> 2. _letter_ ('c') is equal to the label of _node_ ('c'), so set _node_ to the middle child of _node_ ('a') and set _letter_ to the next letter of _cme_ ('m')  
> 3. _letter_ ('m') is greater than the label of _node_ ('a'), but _node_ does not have a right child, so we **failed**
### Insert
To insert a **key**, compare the current character with the **node's label**:
- **If Key Char < Node Label:** Move to the **Left** child. If null, create a new Left child and build a Middle-child "spine" for the remaining characters.
- **If Key Char > Node Label:** Move to the **Right** child. If null, create a new Right child and build a Middle-child "spine" for the remaining characters.
- **If Key Char == Node Label:** **If last character:** Mark current node as a **word-node**.
    - **If not last character:** Move to **Middle** child and next character. If Middle is null, build a "spine" of Middle children for the remaining characters.

```cpp
insert(key): // insert key into this TST
    node = root node of the TST
    letter = first letter of key

    loop infinitely:
        // left child
        if letter < node.label:
            if node has a left child:
                node = node.leftChild

            else:
                node.leftChild = new node labeled by letter
                node = node.leftChild

                iterate letter over the remaining letters of key:
                    node.middleChild = new node labeled by letter
                    node = node.middleChild

                label node as a word-node
                break

        // right child
        else if letter > node.label:
            if node has a right child:
                node = node.rightChild

            else:
                node.rightChild = new node labeled by letter
                node = node.rightChild

                iterate letter over the remaining letters of key:
                    node.middleChild = new node labeled by letter
                    node = node.middleChild

                label node as a word-node
                break

        // middle child
        else:
            if letter is the last letter of key:
                label node as a word-node
                return true

            else:
                if node has a middle child:
                    node = node.middleChild
                    letter = next letter of key
                else:
                    iterate letter over the remaining letters of key:
                        node.middleChild = new node labeled by letter
                        node = node.middleChild

                    label node as a word-node
                    break
```

> [!Example]- Walk Through Example
> Below is the initial example of a **Ternary Search Tree**, and we will demonstrate the process of inserting the word _cabs_:
> 
> ![[Pasted image 20260126154049.png]]
> 1. We start with _node_ as the root node ('c') and _letter_ as the first letter of _cabs_ ('c')
> 2. _letter_ ('c') is equal to the label of _node_ ('c'), so set _node_ to the middle child of _node_ ('a') and set _letter_ to the next letter of _cabs_ ('a')  
> 3. _letter_ ('a') is equal to the label of _node_ ('a'), so set _node_ to the middle child of _node_ ('l') and set _letter_ to the next letter of _cabs_ ('b')  
> 4. _letter_ ('b') is less than the label of _node_ ('l'), but _node_ does not have a left child:
> 5. Create a new node as the left child of _node_, and label the new node with _letter_ ('b')
> 6. Set __node__ to the left child of _node_ ('b') and set _letter_ to the next letter of _cabs_ ('s')
> 7. Create a new node as the middle child of _node_ ('s'), and label the new node with _letter_ ('s')
> 8. Set __node__ to the middle child of _node_ ('s')
> 9. _letter_ is already on the last letter of _cabs_, so label _node_ as a "word node" and we're done!

### Remove
To remove a **key**, use the search logic to locate the target node:
- **If search fails:** The key does not exist; no action is taken.
- **If search succeeds:**
    - Navigate to the node representing the last character of the key.
    - **Action:** Unmark the **word-node** status (e.g., change the color from blue to white).

> [!Note] 
> To optimize memory, you could *recursively* delete nodes that no longer lead to any word-nodes, though simply unmarking is the standard logical removal.

```cpp
remove(key): // remove key if it exists in this TST
    node = root node of the TST
    letter = first letter of key
    loop infinitely:
        // left child
        if letter < node.label:
            if node has a left child:
                node = node.leftChild
            else:
                return                               // key cannot exist in this TST

        // right child
        else if letter > node.label:
            if node has a right child:
                node = node.rightChild
            else:
                return                               // key cannot exist in this TST

        // middle child
        else:
            if letter is the last letter of key and node is a word-node:
                remove the word-node label from node // found key, so remove it from the TST
                return
            else:
                if node has a middle child:
                    node = node.middleChild
                    letter = next letter of key
                else:
                    return                           // key cannot exist in this TST
```

> [!Example]- Walk Through Example
> Below is the initial example of a **Ternary Search Tree**, and we will demonstrate the process of removing the word _mid_:
> 
> ![](https://ucarecdn.com/78791ffc-151c-40b0-a879-306366b27780/)
> 
> 1. We start with _node_ as the root node ('c') and _letter_ as the first letter of _mid_ ('m')
> 2. _letter_ ('m') is greater than the label of _node_ ('c'), so set _node_ to the right child of _node_ ('m')  
> 3. _letter_ ('m') is equal to the label of _node_ ('m'), so set _node_ to the middle child of _node_ ('e') and set _letter_ to the next letter of _mid_ ('i')  
> 4. _letter_ ('i') is greater than the label of _node_ ('e'), so set _node_ to the right child of _node_ ('i') 
> 5. _letter_ ('i') is equal to the label of _node_ ('i'), so set _node_ to the middle child of _node_ ('n') and set _letter_ to the next letter of _mid_ ('d')  
> 6. _letter_ ('d') is less than the label of _node_ ('n'), so set _node_ to the left child of _node_ ('d')  
> 7. _letter_ ('d') is equal to the label of _node_ ('d'), _letter_ is already on the last letter of _mid_, and _node_ is a "word node," so _mid_ exists in the tree
> 8. Remove the "word node" label from _node_

---
## 3. Advanced Features

Like the Multiway Trie, the TST supports ordered operations and prefix-based utilities.

### Alphabetical Iteration

Because TSTs maintain the BST property (Left < Node < Right), an **In-Order Traversal** can retrieve words in sorted order.

```Python
# Pseudocode for Ascending Order
ascendingInOrder(node):
    ascendingInOrder(node.leftChild)
    if node is a word-node:
        output word
    ascendingInOrder(node.middleChild)
    ascendingInOrder(node.rightChild)
```

### Auto-complete

To find all words starting with a prefix:
1. Traverse the TST to the node representing the end of the prefix.
2. Perform an `ascendingInOrder` traversal on that node's **middle child** subtree.

---
## 4. Evaluation for Lexicon ADT

The TST is often the preferred choice for implementing digital lexicons because it balances the trade-offs of other structures.

|**Feature**|**BST (AVL)**|**Multiway Trie**|**Ternary Search Tree**|
|---|---|---|---|
|**Search (Avg)**|$O(\log n)$|$O(k)$|**$O(k + \log n)$**|
|**Space Efficiency**|High|Very Low|**High**|
|**Alphabetical**|Yes|Yes|**Yes**|
|**Auto-complete**|No|Yes|**Yes**|

### Key Takeaways

- **Space over Speed:** TSTs avoid the "wasted pointer" problem of Multiway Tries because each node only has 3 pointers instead of $|\Sigma|$ (e.g., 26 or 256).
- **Balance Matters:** Like a BST, a TST can become "skewed" (unbalanced) if words are inserted in a poor order. Shuffling words before insertion is a common optimization.
- **The "Middle Ground":** It provides the prefix-matching power of a Trie with the memory footprint of a Tree.