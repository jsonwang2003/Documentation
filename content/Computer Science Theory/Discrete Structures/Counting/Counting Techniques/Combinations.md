---
tags:
  - Counting
  - Combinations
  - Permutations
  - Subset
  - Binomial
---
> [!INFO]
> Suppose a set $S$ with $n$ **distinct** elements and you want to know how many ways to form a **subset** of $k$ different elements of $S$.

$$
C(n, k) = \frac{P(n, k)}{k!} = \frac{n!}{k!(n - k)!}
$$

> [!IMPORTANT]
> In combinations, **order does not matter**.
> 
> See more in [[Set Theory]]

---

### Examples

#### 1. Choosing 7 People for a Field Trip

How many ways can I choose 7 people from a class of 50?

$$
C(50, 7)
$$

#### 2. Assigning 7 People to 7 Days

If each of the 7 selected people is assigned to a different day of the week (i.e., order matters):

$$
P(50, 7)
$$

#### 3. Assigning Roles in a Group Project

From a team of 20 people, how many ways can you assign 4 distinct roles (leader, scribe, treasurer, presenter)?

$$
P(20, 4) = 20 \cdot 19 \cdot 18 \cdot 17
$$

#### 4. Selecting 4 People to Attend a Conference

From a team of 20 people, how many ways can you choose 4 to attend a conference (no roles, just selection)?

$$
C(20, 4) = \frac{20 \cdot 19 \cdot 18 \cdot 17}{4 \cdot 3 \cdot 2 \cdot 1}
$$

> [!NOTE]
> Combinations are often contrasted with permutations. See [[rPermutations]] for ordered selections.