---
description: Three levels of specifying an algorithm — high-level (abstractions), mid-level (pseudocode), low-level (concrete implementation) — and why designing top-down helps.
tags:
  - algorithm
  - concepts
aliases:
  - Algorithm Design Levels
  - High/Mid/Low Level Design
---
> [!abstract] Goal 
> While the end goal is to develop fully specified algorithms including all details — data structures, time analysis, etc. — it is useful to come up with a [[#Mid Level Design]] or [[#High Level Design]] version of the algorithm first. This helps set clarity on the logic of the algorithm instead of getting caught up in the details.

---
# High Level Design

A high-level version specifies **what** the algorithm wants to do at every step, in terms of abstractions such as sets, relations, orderings, graphs, etc. — not all the details of **how** it will do these steps.

**Advantages:**

- **Clarity:** presenting a high-level algorithm first gives the reader the main idea of what's going on, before getting caught up in details.
- **Correctness Proofs:** usually much easier to do for high-level algorithms. You just need to make sure the low-level version does what the high-level version specifies.
- **Flexibility:** by showing that the high-level algorithm works, we show that _any_ low-level implementation will solve the problem so long as the data structures and implementation details do what they claim. Then the implementation details can be changed to fit a given situation (dense vs. sparse graphs, memory efficiency vs. time efficiency, parallel vs. sequential) without worry.

---
# Mid Level Design

A mid-level version describes the algorithm in **pseudocode** — more concrete than the high-level, but still abstract enough to be independent of any specific programming language or data structure implementation.

**Advantages:**

- **Precision:** pseudocode forces you to specify the exact sequence of operations, conditions, and loops, catching logical gaps that a high-level description might gloss over.
- **Traceability:** each pseudocode step maps directly to a high-level step, making it easy to verify that the mid-level design faithfully implements the intent of the [[#High Level Design]].
- **Portability:** the pseudocode remains independent of any particular language or library, so it can be translated into Python, Java, C++, etc. without rethinking the logic.
- **Collaboration:** teams can review and agree on the algorithm's logic at this level before committing to implementation details, reducing costly late-stage rewrites.

---
# Low Level Design

A low-level version translates [[#Mid Level Design]] pseudocode into a concrete implementation, specifying exact data structures, language constructs, memory layout, and API calls.

**Advantages:**

- **Details:** if you know exactly how the algorithm will work, a low-level design is preferred.
- **Implementation:** if you need to implement in a programming language, low-level design gives you strict guidelines to translate into actual working code.
- **Time Analysis:** without an implementation or low-level description, there may not be enough detail to determine how many computer steps the procedure will take.

---
# Summary

The three levels form a hierarchy where correctness flows downward: prove the high-level design correct, verify the mid-level faithfully implements it, and confirm the low-level matches the mid-level. This separation also provides flexibility — the high and mid levels can remain stable while low-level details are swapped out to suit different performance constraints, hardware, or languages.

|Level|Specifies|Independent of|Key Advantage|
|---|---|---|---|
|High-Level|What each step does, using abstractions (sets, relations, orderings, graphs)|How those abstractions are implemented|Clarity + easy correctness proofs|
|Mid-Level|Exact sequence of operations, in pseudocode|Programming language / concrete data structures|Precision + portability|
|Low-Level|Concrete data structures, language constructs, memory layout|Nothing — this is the actual implementation|Enables real time/space analysis|

---
# References / Links

- [[Explore]]