---
title: Machine Learning Systems
---
Machine Learning Systems (ML Systems) focus on the principles and practices of engineering artificially intelligent systems at scale. This folder explores the intersection of distributed systems, hardware acceleration, and software engineering required to manage the lifecycle of an AI application.

---
## Part I: Systems Foundations
This section establishes the theoretical and technical background required to understand system-level decisions.
- **Chapter 1: [[Machine Learning Systems/Introduction/index|Introduction]]** – The engineering revolution in AI, defining ML systems, and the "Bitter Lesson" of why systems engineering matters.
- **Chapter 2: ML Systems** – Exploring the deployment spectrum, from maximizing power in the Cloud to resource-constrained TinyML and Hybrid architectures.
- **Chapter 3: DL Primer** – The engineering foundations of Deep Learning, including mathematical translations of neural concepts and production inference pipelines.
- **Chapter 4: DNN Architectures** – System implications and computational mapping for various network types (CNNs, RNNs, and Transformers).

---
## Part II: Design Principles
This section focuses on the "how" of building reliable systems through systematic frameworks and engineering disciplines.
- **Chapter 5: AI Workflow** – Understanding the six core lifecycle stages of ML development and how decisions cascade through a system.
- **Chapter 6: Data Engineering** – Data engineering as a systems discipline, focusing on the four pillars of quality, reliability, scalability, and governance.
- **Chapter 7: AI Frameworks** – Analysis of framework abstractions (Computational Graphs, Tensors) and selection methodologies for platforms like TensorFlow, PyTorch, and JAX.
- **Chapter 8: AI Training** – The evolution of training systems, including optimization algorithms, mixed-precision training, and distributed systems (Data/Model Parallelism).

---
## Part III: Performance Engineering
This section examines the optimization of systems to meet real-world constraints and scaling laws.
- **Chapter 9: Efficient AI** – The efficiency imperative, compute-optimal resource allocation, and the trade-offs between algorithmic, compute, and data efficiency.
- **Chapter 10: Model Optimizations** – Structural methods like pruning, knowledge distillation, and quantization to balance performance and deployment context.
- **Chapter 11: Tensor Compilers** – Strategies for mapping neural networks to hardware and the role of compiler support in the ML pipeline.
- **Chapter 12: Hardware Accelerators** – Deep dive into the specialized silicon (GPUs, TPUs, FPGAs, ASICs) that powers modern AI.

---
## Part IV: Robust Deployment
This final section addresses the complexities of maintaining and evolving AI systems in production environments.
- **Chapter 13: ML Operations (MLOps)** – Managing technical debt, boundary erosion, and system complexity in production.
- **Chapter 14–19: Specialized Deployment** – Detailed explorations of Cloud, Edge, Mobile, and TinyML systems, as well as AI Safety and Security.
- **Chapter 20: AGI Systems** – Exploring intelligence as a systems problem, the scaling hypothesis, and compound AI systems frameworks.

---
## Key Learning Outcomes

The goal of these notes is to move beyond individual components to develop **systems thinking**—the ability to reason about latency vs. accuracy trade-offs and recognize how infrastructure choices propagate through an entire architecture.

---

**Source:** Notes based on Janapa Reddi, V. (2025). _Introduction to Machine Learning Systems_. [mlsysbook.ai](https://mlsysbook.ai)