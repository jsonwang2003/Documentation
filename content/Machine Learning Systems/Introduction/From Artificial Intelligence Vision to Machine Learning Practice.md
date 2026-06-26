### Artificial Intelligence (The Vision)

**Artificial Intelligence** represents the broad goal of creating systems capable of performing tasks that typically require human-like intelligence.
- **Core Capabilities:** Recognizing images, understanding language, making decisions, and solving problems.
- **Definition:** The goal of creating systems that can perform tasks requiring human-like intelligence, including _learning_, _reasoning_, and _adapting to new situations_.
### Machine Learning (The Methodology)
**Machine Learning** represents the practical discipline and methodological approach used to achieve AI.
- **Mechanism:** Rather than following predetermined rules, ML systems build on mathematical foundations to automatically discover patterns in data.
- **Definition:** The practical approach to achieving AI by building systems that discover patterns in data through _computational techniques_ rather than following hard-coded logic.

---
## Case Study: The Evolution of Chess Systems

The goal (AI) remains constant—"Play chess at a human level"—but the engineering approach has undergone a fundamental shift.

|**Approach**|**Symbolic AI (Pre-ML)**|**Machine Learning**|
|---|---|---|
|**Logic**|Hand-crafted rules (e.g., "control the center").|Statistical analysis of millions of games.|
|**Development**|Experts encode thousands of principles.|System discovers its own winning strategies.|
|**Nature**|**Brittle:** Fails in unanticipated scenarios.|**Adaptive:** Learns from data outcomes.|

---
## The Paradigm Shift: From Rules to Systems

The transition from **Symbolic Reasoning** to **Statistical Learning** represents a major paradigm shift[^1] in how intelligence is constructed.
- **Rule-Based Scaling:** Limited by human programmer effort. Adding "intelligence" requires manually encoding every new rule.
- **Data-Driven Scaling:** Scales with compute and data infrastructure. Performance improves by adding more GPUs and training data rather than more programmers.7

> [!ABSTRACT] Machine Learning is the Dominant Approach
> 
> Modern ML requires systems to collect, process, and learn from data at massive scale.8 Because success now depends on infrastructure—training billion-parameter models and serving predictions globally—systems engineering has become the critical bottleneck for AI advancement.

---
## Applied Learning Models

Modern systems acquire capabilities through high-volume data exposure:
- **Object Recognition:** Mirrors human visual learning; requires exposure to numerous examples to develop robust recognition.
- **Natural Language Processing (NLP):** Acquires linguistic capabilities through extensive statistical analysis of textual data.

[^1]: **Paradigm Shift**: A term coined by Thomas Kuhn (1962).12 In AI, this shift moved the burden of "knowledge" from the programmer to the system. This had profound systems implications: success shifted from human expert knowledge to the ability to build infrastructure that can handle the bandwidth, fault tolerance, and consistency required for petabyte-scale training.