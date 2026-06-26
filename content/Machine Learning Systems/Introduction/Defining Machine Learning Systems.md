## Definition: Machine Learning System

> [!INFO] Definition
> 
> A Machine Learning System is an integrated computing system comprising three core components:
> 
> 1. **Data:** Guides algorithmic behavior.
>     
> 2. **Learning Algorithms:** Extract patterns from this data.
>     
> 3. **Computing Infrastructure:** Enables both the **learning process (training)** and the **application of applied knowledge (inference/serving)**.
>     
> 
> These components create a computing system capable of making _predictions_, generating _content_, or taking _action_ based on learned patterns.

> [!NOTE]
> 
> No universally accepted definition of machine learning systems exists. This definition adopts a perspective that encompasses the entire ecosystem in which algorithms operate.

---
## Component Interdependencies (The AI Triangle)

The core of any machine learning system consists of three interrelated components that form a triangular dependency. Each element shapes the possibilities of the others:
- **Models/Algorithms:** Mathematical methods that learn patterns. The architecture dictates the computational demands and the structure of data required.
- **Data:** Processes for collecting, storing, and managing information. The scale and complexity of data influence the required infrastructure and determine which model architectures are feasible.
- **Computing Infrastructure:** Hardware/software (GPUs, TPUs) that enables operations at scale. Infrastructure establishes the practical limits on both model scale and data processing capacity. In the context of ML systems, it plays the role of providing the necessary resources for both training and inference

![[Pasted image 20251226114722.png]]

> [!NOTE] **Design Principle**
> 
> Machine learning system performance relies on the coordinated interaction of these elements. Effective system design requires balancing these interdependencies to optimize overall performance and feasibility; limitations in one component inevitably constrain the others.

> [!Question] **Consider a scenario where an ML system's data component is limited by storage capacity. How might this affect the other components of the system?**
> If the data is limited by storage capacity, it may restrict the volume and variety of the data available for training, potentially leading to less effective learning algorithms. Additionally, the computing infrastructure may be underutilized if it cannot process larger datasets. This independency emphasizes the need for balanced system design to optimize overall performance.

---
## Professional Analogy: The Space Exploration Framework

To understand the orchestration required for ML systems, consider the roles within a space mission:

|**Role**|**ML System Component**|**Function**|
|---|---|---|
|**Astronauts**|**Algorithm Developers**|Exploring new frontiers and making discoveries.|
|**Mission Control**|**Data Science Teams**|Ensuring the constant flow of critical information and resources.|
|**Rocket Engineers**|**Infrastructure Engineers**|Designing and building the systems that enable the mission.|

Just as space missions require the seamless integration of personnel and hardware, machine learning systems demand careful orchestration of algorithms, data, and computing infrastructure.

---
## Historical Case Study: AlexNet and Co-Design

The 2012 AlexNet[^1] breakthrough illustrates the principle of **hardware-software co-design** that defines modern ML systems engineering.

The "ImageNet moment" succeeded because the **algorithmic innovation** (Convolutional Neural Networks) perfectly matched the **hardware capability** (Parallel GPU architectures).
- **Parallelism:** Convolutional operations are inherently parallel, making them naturally suited to a GPU’s thousands of cores.
- **Repurposing:** GPUs originally designed for gaming were repurposed to provide 10-100x speedups over traditional CPUs for ML tasks.

This co-design approach—where software is built to exploit specific hardware strengths—continues to shape ML system development across the industry.

[^1]: **AlexNet**: A breakthrough deep learning model that won the 2012 ImageNet competition, reducing top-5 error rates from 26.2% to 15.3%. It proved that with enough data (1.2 million images), computing power (two GPUs for 6 days), and clever engineering (dropout, data augmentation), neural networks could achieve superhuman performance on complex visual tasks.