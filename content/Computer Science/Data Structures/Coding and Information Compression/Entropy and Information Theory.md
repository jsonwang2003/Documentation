## 1. Data vs. Information

To understand compression, we must distinguish between the physical representation and the conceptual meaning.

- **Data:** The raw representation of information (e.g., numbers, symbols, measurements).
- **Information:** The actual content or "message" extracted from that data.

> [!Example]
> A grade of "100%" is **data**. The **information** derived is "this student succeeded in the course."

---
## 2. Entropy: The Measure of Predictability

In Information Theory, **Shannon entropy** measures the unpredictability of information content.

- **High Entropy (Non-Uniform):** When symbols appear with balanced frequencies (e.g., `ACGT`). This is "unpredictable" and carries more information per character.
- **Low Entropy (Uniform):** When one symbol dominates (e.g., `AAAA`). This is "predictable" and carries very little information.

### The Golden Rule of Compression:

$$
\text{More Uniform Data} \to \text{Less Entropy} \to \text{ Less Information stored in the Data}
$$

---
## 3. Optimizing Storage: Fixed vs. Variable Length

The goal of compression is to make the **memory used** as close to the **information content** as possible.

### A. Fixed-Length Encoding (Naive)

By default, text files use **8 bits (1 byte)** per character (ASCII).

- **DNA Example:** Since DNA has only 4 letters, we only need 2 bits (log2​(4)=2) to represent each.
- **A → 00, C → 01, G → 10, T → 11.**
- **Result:** A guaranteed 4-fold reduction in file size (from 8 bits to 2 bits per letter).
### B. Variable-Length Encoding (Smart)

We can do better by using our knowledge of character frequency. We assign **shorter codes** to the most frequent characters and **longer codes** to rare ones.

- **Scenario:** A message where **A** is very common and **T** is rare.
- **Mapping:** A → `0`, C → `10`, G → `110`, T → `111`.
- **Trade-off:** We lose efficiency on `T`, but because `A` appears so much more often, the _average_ bits per character drops significantly.

---
## 4. The Overhead Cost

When using frequency-based encoding, we cannot assume the recipient knows our mapping. We must include a **header** at the start of the file that contains the frequency information or the coding tree.

- **Large Files:** The overhead is negligible compared to the massive savings.
- **Small Files:** The overhead might actually make the "compressed" file larger than the original.

---
### Key Takeaway

Entropy defines the **theoretical limit** of how much we can compress data. If a message has 1.6 bits of entropy per character, we can never safely compress it to 1 bit per character without losing information.