> [!Abstract] Goal
> While the end goal is the develop fully specified algorithms including all details, such as:
> - Data Structures
> - Time Analysis
> - etc.
> 
> It is useful to come up with a [[#Mid Level Design]] or  [[#High Level Design]] version of the algorithm first. This helps setting clarity on the logic of the algorithm instead of getting caught up in the details.

---
# High Level Design
A high-level version specifies **what** the algorithm wants to do at every step, in terms of abstractions such as *sets*, *relations*, *orderings*, *graphs*, etc. Not all of the details of **how** it will do these steps 
## Advantages
- **Clarity**: Presenting a high-level algorithm first gives the reader the main idea of what's going on, before getting caught up in details
- **Correctness Proofs**: Usually much easier to do for high-level algorithms. Just need to make sure that the low-level version does what the high-level version specifies
- **Flexibility**: By showing that the high-level algorithm works, we show that **any** low-level implementation will solve the problem so long as the data structures and implementation details do what they claim. Then we can change the details of the implementation to fit a given situation (**dense vs. sparse graphs**, **memory efficiency vs. time efficiency**, **parallel vs. sequential**) without worries
---
# Mid Level Design
A mid-level version describes the algorithm in pseudocode, more concrete than the high-level, but still abstract enough to be independent of any specific programming language or data structure implementation
## Advantages
- **Precision**: Pseudocode forces to specify the exact sequence of operations, conditions, and loops, catching logical gaps that a high-level description might gloss over
- **Traceability**: Each pseudocode step maps directly to a high-level step, making it easy to verify that the mid-level design faithfully implements the intent of the [[#High Level Design]]
- **Portability**: The pseudocode remains independent of any particular language or library, so it can be translated into Python, Java, C++, etc. without rethinking the logic
- **Collaboration**: Teams can review and agree on the algorithm's logic at this level before committing to implementation details, reducing costly late-stage rewrites
---
# Low Level Design
A low-level version translates [[#Mid Level Design]] pseudocode into concrete implementation, specifying exact data structures, language constructs, memory layout, and API calls.
## Advantages
- **Details**: If you know exactly how the algorithm will work, a low-level design is preferred
- **Implementation**: If you need to **implement in a programming language**, low-level design gives you strict guidelines to translate into an actual working computer code
- **Time Analysis**: Without an implementation or low-level description, there may not be enough details to determine **how many computer steps** the procedure will take

---
# Summary
The three levels form a hierarchy where correctness flows downward: prove the high-level design correct, verify the mid-level faithfully implements it, and confirm the low-level matches the mid-level. This separation also provides flexibility — the high and mid levels can remain stable while low-level details are swapped out to suit different performance constraints, hardware, or languages.
