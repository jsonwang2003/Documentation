Thus far in the text, we have only discussed _storing keys_ in a data structure, and we have discussed numerous data structures that can be used to find, insert, and remove keys (as well as their respective trade-offs). Then, throughout this chapter, we introduced the **Hash Table**, a data structure that, on average, performs _extremely fast_ (i.e., constant-time) find, insert, and remove operations.

However, what if we wanted to push further than simply storing _keys_? For example, if we were to teach a class and wanted to store the students' names, we could represent them as strings and store them in a **Hash Table**, which would allow us to see if a student is enrolled in our class in constant time. However, what if we wanted to _also_ store the students' grades? In other words, we can already query a student's name against our **Hash Table** and receive "true" or "false" if the student is in our table or not, but what if we want to instead return the student's grade?

The functionality we have described, in which we query a _key_ and receive a _value_, is defined in the **Map ADT**, which we will formally describe very soon. After discussing the **Map ADT**, we will introduce a data structure that utilizes the techniques we used in **Hash Tables** to _implement_ the **Map ADT**: the **Hash Map**.

**Sad Fact:** Most people do not know the difference between a **Hash Table** and a **Hash Map**, and as a result, they use the terms interchangeably, so beware! As you will learn, though a **Hash Map** is built on the same _premise_ as a **Hash Table**, it is a bit different and arguably more convenient in day-to-day programming.

**Note:** If you have experience programming in Python and have used a [Python dictionary](https://docs.python.org/2/tutorial/datastructures.html#dictionaries), then congratulations! You have been using a **Hash Map** all along and are already one step (no pun intended) ahead of us! If you have no clue what we are talking about, then read on!

---
As we mentioned in the previous step, our goal is to have some efficient way to store students and their grades such that we could query our data structure with a student's name and then have it return to us their grade. We could, of course, store the student names and grades in separate **Hash Tables**, but how would we know which student has which grade?

![](https://ucarecdn.com/23f88623-1ffd-46d2-9e28-97739e0a9804/)

Consequently, we want to find a way to be able to store a student's name _with_ his or her letter grade. This is where the **Map Abstract Data Type** comes into play! The **Map ADT** allows us to _map_ keys to their corresponding values. The **Map ADT** is often called an _associative array_ because it gives us the benefit of being able to _associatively_ cluster our data.

Formally, the **Map ADT** is defined by the following set of functions:

- **put(key,value):** perform the insertion, and return the previous _value_ if overwriting, otherwise NULL
- **get(key):** return the value associated with _key_ if _key_ is in the **Map**, otherwise fail
- **remove(key):** remove the (_key_, _value_) pair associated with key, and return _value_ upon success or NULL on failure
- **size():** return the number of (_key_, _value_) pairs currently stored in the **Map**
- **isEmpty():** return true if the **Map** does not contain any (_key_, _value_) pairs, otherwise return false

We have now formally defined the **Map** **ADT**, but how can we go about actually implementing it?

---
The **Map ADT** can theoretically be implemented in a multitude of ways. For example, we could implement it as a **Binary Search Tree**: we would store _two_ items inside each node, the _key_ and the _value_, and we would keep the **Binary Search Tree** ordering property based on just _keys_.

_However_, if we didn't care so much about the sorting property but rather wanted faster _put_ and _has_ operations (e.g. constant time), then the **Map ADT** could also be implemented effectively as a **Hash Table**: we refer to this implementation as a **Hash Map**.

Implementation-wise, a **Hash Map** has the following set of operations:

- **insert(key,value):** perform the insertion, and return the previous _value_ if overwriting, otherwise NULL
- **find(key):** return the _value_ associated with the _key_
- **remove(key):** remove the (_key_, _value_) pair associated with key, and return _value_ upon success or NULL on failure
- **hashFunction(key):** return a hash value for _key_, which will then be used to map to an index of the backing array
- **key_equality(key1, key2):** return true if _key1_ is equal to _key2_, otherwise return false
- **size():** return the number of (_key_, _value_) pairs currently stored in the **Hash Map**
- **isEmpty():** return true if the **Hash Map** does not contain any (_key_, _value_) pairs, otherwise return false

Just like a **Hash Table**, a **Hash Map** uses a **hash function** for the purpose of being able to access the addresses of the tuples inserted. Consequently, in a **Hash Map**, keys must be hashable and have an associated equality test to be able to check for uniqueness. In other words, to use a custom class type as a key, one would have to overload the hash and equality member functions.

---
When we find, insert, or remove (_key_, _value_) pairs in a **Hash Map**, we do _everything exactly like_ we did with a **Hash Table**, but with respect to the _key_.

For example, in the **Hash Map** **insertion** algorithm, because we are given a (_key_, _value_) pair, we hash only the _key_ but store the _key_ and the _value_ together. Below is an example of a **Hash Map** that contains multiple (_key_, _value_) pairs:

![](https://ucarecdn.com/e9108a34-1aad-4123-b52d-be6e9d695028/ "Image: https://ucarecdn.com/e9108a34-1aad-4123-b52d-be6e9d695028/")

When we want to **find** elements, we perform the exact same "find" algorithm as we did with a **Hash Table**, but again with respect to the _key_ (which is why our "find" function only had _key_ as a parameter, not _value_). Once we find the (_key_, _value_) pair, we simply return the _value_. For example, in the example **Hash Map** above, if we want to perform the "find" algorithm on "Kammy," we perform the regular **Hash Table** "find" algorithm on "Kammy." When we find the pair that has "Kammy" as its _key_, we return the _value_ (in this case, 'A').

Just like with finding elements, if want to **remove** elements from our **Hash Map**, we perform the **Hash Table** "remove" algorithm with respect to the _key_ (which is why our "remove" function only had _key_ as a parameter, not _value_), and once we find the (_key_, _value_) pair, we simply remove the pair.

---
In case the previous step was too "hand-wavy" with regard to how we go about inserting elements, let's look at the pseudocode for the **Hash Map** operations below. Note that a **Hash Map** can be implemented using a **Hash Table** with any of the collision resolution strategies we discussed previously in this chapter. In all of the following pseudocode, the **Hash Map** is backed by an array arr, and for a (_key_, _value_) pair pair, pair.key indicates the _key_ of pair and pair.value indicates the _value_ of pair.

In the **insert** operation's pseudocode below, we ignore collisions (i.e., each key maps to a unique index in the backing array) because the actual insertion algorithm would depend on which collision resolution strategy you choose to implement.

```cpp
insert(key,value): // insert <key,value>, replacing old value with new value if key exists
    index = hashFunction(key)
    returnVal = NULL

    // if key already exists, save the old value
    if arr[index].key == key:
        returnVal = arr[index].value // we want to return the old value instead of NULL

    // perform the insertion
    arr[index] = <key,value>
    return returnVal
```

With respect to insertion, originally, in a **Hash Table**, if a key that was being inserted already existed, we would abort the insertion. In a **Hash Map**, however, attempting to insert a key that already exists will _not_ abort the insertion. Instead, it will result in the original value being overwritten by the new one.

The pseudocode for the **find** operation of a **Hash Map** is provided below. Note that this "find" algorithm returns the _value_ associated with _key_, as opposed to a Boolean value as it did in the **Hash Table** implementation.

```cpp
find(key): // return value associated with key if key exists, otherwise return NULL
    index = hashFunction(key)
    if arr[index].key == key:
        return arr[index].value
    else:
        return NULL
```

The pseudocode for the **remove** operation of a **Hash Map** is provided below. Just like with the pseudocode for the insertion algorithm above, in the pseudocode below, we ignore collisions (i.e., each key maps to a unique index in the backing array) because the actual remove algorithm would depend on which collision resolution strategy you choose to implement.

```cpp
remove(key): // remove <key,value> if key exists and return value, otherwise return NULL
    index = hashFunction(key)
    returnVal = NULL

    // if key already exists, save the old value
    if arr[index].key == key:
        returnVal = arr[index].value // we want to return the value instead of NULL
        delete arr[index]            // perform the removal

    // return the appropriate value
    return returnVal
```

---
In practice, however, we realize that you will more often than not be using the built-in implementation of a **Hash Map** as opposed to implementing it from scratch, so how do we use C++'s **Hash Map** implementation?

In C++, the implementation of a **Hash Map** is the [unordered__map_](http://www.cplusplus.com/reference/unordered_map/unordered_map/), and it is implemented using the **Separate Chaining** collision resolution strategy. Just to remind you, in C++, the implementation of a **Hash Table** is the [unordered__set_](http://www.cplusplus.com/reference/unordered_set/unordered_set/).

Going all the way back to the initial goal of implementing a grade book system, the C++ code to use a **Hash Map** would be the following:

```cpp
unordered_map<string, string> gradeBook = {
                { "Kammy", "A"},
                { "Alicia", "C"},
                { "Anjali", "D"},
                { "Nadah", "A"}
};
```

If we wanted to add a new student to our grade book, we would do the following:

```cpp
gradeBook.insert({"Bob", "B"});

/* Our new hash map would look something like this:
             {  { "Kammy", "A"},
                { "Bob" , "B"},
                { "Alicia", "C"},
                { "Anjali", "D"},
                { "Nadah", "A"}  };
  
   Note how there is no ordering property, as expected */
```

If we wanted to check Nadah's grade in our grade book, we would do the following:

```
cout << gradeBook["Nadah"] << endl; // [] operator returns the value stored at the key

/* Output:
   A
*/
```

Although we have mentioned many times that there is no particular ordering property when it comes to a **Hash Map** (as well as a **Hash Table**), we can still _iterate_ through the inserted objects using a for-each loop like so:

```cpp
for (auto student : gradeBook) {

   cout << student.first << ": " << student.second << endl; // .first returns the key
                                                            // .second returns the value
}

/* Output:
   Bob: B
   Kammy: A
   Anjali: D
   Alicia: C
   Nadah: A
*/
```

  
**Note**: C++ deviates slightly from the traditional **Map** **ADT** with respect to insertion. When inserting a duplicate element in the C++ unordered_map, the original value is not replaced. On the other hand, Python's implementation of the **Map** **ADT** does in fact replace the original value in a duplicate insertion.

---
It is also important to note that, in practice, we often use a **Hash Map** to implement "one-to-many" relationships. For example, suppose we want to implement an "office desk" system in which each desk drawer has a different label: "pens," "pencils," "personal papers," "official documents," etc. Inside each particular drawer, we expect to find office items related to the label. In the "pens" drawer, we might expect to find our favorite black fountain pen, a red pen for correcting documents, and that pen we "borrowed" from our friend months ago.

How would we use a **Hash Map** to implement this system? Well, the _drawer labels_ would be considered the _keys_, and the _drawers_ with the objects inside them would be considered the corresponding _values_.

The C++ code to implement this system would be the following:

```cpp
unordered_map<string, vector<OfficeSupply>> desk = {
                { "pens", {favPen, redPen, stolenPen} },
                { "personal papers", {personalNote}   }
};
```

In the **Hash Map** above, we are using _keys_ of type string and _values_ of type vector`<OfficeSupply>` (where OfficeSupply is a custom class we created). Note that the _values_ inserted into the **Hash Map** are **NOT** OfficeSupply objects, but vectors of OfficeSupply objects.

If we now wanted to add a printed schedule to the "personal papers" drawer of our desk, we would do the following:

```cpp
desk["personal papers"].push_back(schedule); 

/*
   desk["personal papers"] returns the {personalNote} vector
   push_back adds a schedule object of type OfficeSupply to the {personalNote} vector

   our hash map now looks like this:
   {
       { "pens", {favPen, redPen, stolenPen} },
       { "personal papers", {personalNote, schedule} }
   }
*/
```

**Note:** If we wanted to calculate the worst-case time complexity of finding an office supply in our desk, we would now need to take into account the time it takes to find an element in an unsorted vector (which is an Array List) containing _n_ OfficeSupply objects, which would be O(_n_). If we wanted to ensure constant-time access across OfficeSupply objects, we could also use a **Hash Table** instead of a vector (yes, we are saying that you can use unordered_set objects as _values_ in your unordered_map).

To easily output which pens we have in our desk, we could use a for-each loop like so:

```cpp
for (auto pen : desk["pens"]) {
   cout << pen << endl;
}
/* Output:
   favPen
   redPen
   stolenPen
*/
```

---
We began this chapter with the motivation of obtaining _even faster_ find, insert, and remove operations than we had seen earlier in the text, which led us to the **Hash Table**, a data structure with **O(1)** find, insert, and remove operations in the **average case**. In the process of learning about the **Hash Table**, we discussed various properties and design choices (both in the **Hash Table** itself as well as in **hash functions** for objects we may want to store) that can help ensure that we actually experience the constant-time performance on average.

We then decided we wanted even _more_ than simply _storing_ elements: we decided we wanted to be able to _map_ objects to other objects (map _keys_ to _values_, specifically), which led us to the **Map ADT**. Using our prior knowledge of **Hash Tables**, we progressed to the **Hash Map**, an extremely fast implementation of the **Map** **ADT**.

In practice, the **Hash Table** and the **Hash Map** are arguably two of the most useful data structures you will encounter in daily programming: the **Hash Table** allows us to store our data and the **Hash Map** allows us to easily cluster our data, both with great performance.