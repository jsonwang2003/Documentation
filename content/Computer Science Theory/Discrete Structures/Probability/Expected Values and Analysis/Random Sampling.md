> [!ABSTRACT]
> 
> Random sampling is a technique used to select a subset of individuals or outcomes from a larger population. The goal is to ensure that the sample accurately represents the characteristics of the whole, often by ensuring every outcome has a known or equal probability of being chosen.

---
## Uniform Random Sampling

> [!INFO]
> 
> A Uniform Random Sampling of a sample space $S$ is an experiment that yields one outcome from $S$ such that each outcome of $S$ is equally likely to occur.
### Simulating Distributions
Uniform sampling can be used to simulate different probability distributions by partitioning the sample space.
- **Simulating a Coin with a Die**: To generate $H$ and $T$ with $P = 1/2$ using a six-sided die, group the outcomes into two equal sets:
    - $\{1, 2, 3\} \to \text{Heads}$
    - $\{4, 5, 6\} \to \text{Tails}$
- **Simulating a Die with a Coin**: This requires multiple flips to create a large enough sample space (at least $2^3 = 8$ outcomes) to cover the 6 faces of a die. Outcomes that do not map to a die face are handled via **Rejection Sampling**.

![[Pasted image 20251019110124.png]]

---
## Rejection Sampling
Rejection sampling is a procedure used to simulate a selection from a subset $T \subseteq S$ (uniformly at random) when you only have the means to select from the larger set $S$.
### The Procedure
1. **Select** an outcome $x$ from the larger set $S$ uniformly.
2. **Verify**: If $x \in T$, keep it as your result.
3. **Reject**: If $x \notin T$, discard the result and return to Step 1.

> [!NOTE]
> 
> This method is highly effective for simulating specific distributions (like a 6-sided die) using binary sources (like a coin). You generate enough bits to exceed the target range and "reject" any value outside that range.

---
## Related Notes
- **[[Counting and Probability]]**: The foundation for determining the size of sample spaces $S$ and $T$.
- **[[Uniform vs Non-Uniform Distributions]]**: Understanding the difference between equally likely outcomes and weighted sampling.
- **[[Expected Value]]**: Used to calculate how many "trials" on average it takes for Rejection Sampling to succeed.
- **[[Geometric Distribution]]**: The number of attempts until a successful sample is kept in Rejection Sampling follows a geometric distribution.