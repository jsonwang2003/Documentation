## Transformation of Software Systems

| Traditional Software Architectures                               | Machine Learning Systems                                       |
| ---------------------------------------------------------------- | -------------------------------------------------------------- |
| **Explicitly Programmed:** Logic is hard-coded by developers.    | **Probabilistic:** Behaviors emerge from statistical patterns. |
| **Predictable Failure:** Fails via observable crashes or errors. | **Silent Degradation:** Fails via "silent" accuracy erosion.   |
| **Deterministic Outputs:** Same input yields same path.          | **Autonomous Adaptation:** Learned behaviors evolve over time. |

This transformation introduces the core challenges that define the discipline of **Machine Learning Systems Engineering**:
- **Reliability:** Ensuring consistent performance in systems whose behaviors are learned rather than programmed.
- **Scalability:** Achieving efficiency for systems processing **petabyte-scale[^1]** datasets while serving billions of concurrent users.
- **Robustness:** Maintaining performance when operational data distributions diverge from training distributions (Distribution Shift).

---
## Historical Context: The Three Inflection Points

Engineering practice has reached its third major historical milestone:
1. **Industrial Revolution:** Established mechanical engineering to manage physical forces.
2. **Digital Revolution:** Formalized computational engineering to handle algorithmic complexity.
3. **AI Engineering Revolution:** Necessitates a paradigm for systems exhibiting learned behaviors and autonomous adaptation at scales exceeding traditional methodologies.

---
## The AI Triangle: Interdependency

The performance of an ML system is constrained by the coordinated interaction of three components. Limitations in any one—such as the **Memory Wall** (the gap between processor speed and memory bandwidth)—directly constrain the capabilities of the others.
- **Algorithms:** Mathematical models that learn patterns.
- **Data:** Infrastructure for collecting, storing, and managing information.
- **Computing Infrastructure:** The hardware/software stack (GPUs, TPUs) enabling training and serving.

> [!Question] How does the AI Triangle framework help in understanding machine learning systems?
> > [!Example]
> The AI Triangle framework helps understand machine learning systems by illustrating the interdependencies among data, algorithms, and computational infrastructure. For example, changes in data quality can affect algorithm performance, which in turn impacts infrastructure requirements. This is important because it guides the design and optimization of ML systems.

---
## The Five-Pillar Framework of AI Engineering

The discipline is partitioned into five interconnected sub-disciplines that address the unique challenges of ML systems:
1. **Data Engineering:** Building robust pipelines to ensure quality and handle distribution shifts.
2. **Model Training:** Orchestrating large-scale distributed computation across thousands of GPUs.
3. **Model Deployment:** Creating reliable infrastructure to serve models from the cloud to the edge.
4. **Operation & Maintenance:** Implementing specialized monitoring to catch silent performance degradation.
5. **Ethics & Governance:** Integrating fairness, privacy, and safety into the system lifecycle.

---
## Sutton's 'Bitter Lesson' & The Scaling Hypothesis

Current progress is driven by the insight that larger neural networks trained on more data using more compute consistently solve increasingly complex tasks.

**The Bitter Lesson:** 
	Domain general computational methods ultimately supersede hand-crafted knowledge representations, positioning systems engineering as central to AI advancement.

> [!EXAMPLE]
> deep learning has surpassed symbolic AI in many tasks. This is important because it underscores the shift towards systems engineering as central to AI advancement.

[^1]: One petabyte equals 1,000 terabytes or roughly 1 million gigabytes. The engineering challenge involves the bandwidth, fault tolerance, and consistency guarantees needed to make these datasets useful for training and inference.