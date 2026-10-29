# AGENTS.md

This file provides context about this Obsidian vault for AI agents.

## Vault Overview

This vault serves as a comprehensive academic repository and personal knowledge base for a computer science, artificial intelligence, and data science curriculum (specifically aligned with coursework at UCSD). It spans foundational mathematical and theoretical CS, low-level computer systems and architecture, algorithms, machine learning systems, and interview preparation.

With over 700 markdown notes, the vault functions both as an active study and course-tracking hub (including lecture notes, coursework exercises, and problem sets) and as an enduring reference library of deeply decomposed technical concepts.

## Organization

The vault is organized hierarchically into academic and technical disciplines using strictly categorized topic folders. Subfolders break down broad subjects into granular conceptual components and coursework modules:

- **Computer Systems (252 files)**: Major focus area covering Operating Systems (kernel, concurrency, memory management, file systems), System Programming (extensive coursework notes, memory management, processes), and Digital Systems (ALU, logic design, RTL, sequential circuits).
- **Computer Science Introduction (133 files)**: Foundational programming languages (C, C++), Data Structures (trees, hashing, string searching), and Algorithms (divide and conquer, dynamic programming, graph algorithms, greedy techniques).
- **Computer Science Theory (105 files)**: Deep dives into Discrete Structures (counting, discrete probability, algorithms) and Theory of Computability (automata, formal languages, complexity, proofs).
- **Artificial Intelligence & Data Science (~140 files)**: Machine Learning (supervised, unsupervised, deep learning, RL), Ethical AI & Policy, ML Systems, Model Evaluation, and Statistical Inference.
- **Supplementary Areas**: `leetcode/` for algorithmic problem-solving practice, `Development Environemnt/` (GDB, Valgrind, Git, shell), `Mathematics/`, `Physics/`, `Robotics/`, and `Amazon Web Service/`.

Almost every directory contains an `index` file functioning as a Map of Content (MOC) to link subtopics together. Standardized note creation is supported via `_templates/` (e.g., `index template`, `Computability Theory Template`, and `Discrete Structures Template`). Visual diagrams are integrated via `Excalidraw/`.

## Key Topics

- **Computer Systems & Architecture**: Operating Systems (virtual memory, concurrency, kernels, file systems), Systems Programming in C/C++, and Digital Logic Design (ALU, RTL, circuits).
- **Computer Science Theory**: Discrete Structures (combinatorics, counting rules, discrete probability, graph theory) and Theory of Computability (finite automata, formal languages, Turing machines, complexity classes).
- **Algorithms & Data Structures**: Dynamic programming, graph traversals, divide-and-conquer, hashing, tree structures, and LeetCode algorithmic problem solving.
- **Artificial Intelligence & Machine Learning**: Deep learning architectures, reinforcement learning, supervised/unsupervised learning, machine learning systems engineering, and AI ethics/policy.
- **Data Science & Statistics**: Statistical inference, regression analysis, descriptive analytics, classification, clustering, and model evaluation.
- **Developer Tooling & Systems**: Debugging utilities (GDB, Valgrind, GPROF), Unix shell scripting, Git version control, cloud computing (AWS), and robotics mapping.

## User Preferences

The user maintains an exceptionally structured, rigorous, and syllabus-aligned curriculum vault. Notes are fine-grained, with individual notes dedicated to single concepts, rules, or theorem definitions (e.g., individual counting rules like quotient rule or sum rule).

Responses should be technically rigorous, formal, and precise. Use standard mathematical and computer science notation (definitions, theorems, proofs, pseudocode, time/space complexity analyses) where appropriate. Avoid hand-wavy explanations or overly casual language.

## Custom Instructions

- **Maintain MOCs (`index` files)**: When adding, renaming, or refactoring notes within a folder, ensure the corresponding directory `index` note is updated with the link to preserve navigation.
- **Use Domain Templates**: When generating new notes, check `_templates/` for relevant templates (`Computability Theory Template`, `Discrete Structures Template`, `index template`) to adhere to established frontmatter and section layouts.
- **Respect Granular Separation**: Follow the vault's established pattern of creating focused, bite-sized conceptual notes rather than massive monolithic files, especially in theoretical and systems topics.
- **Technical Rigor & Code Standards**: When generating code examples, prioritize low-level safety and accuracy (specifically for C/C++, GDB, POSIX system calls, and Linux memory management).
- **Link Preservation**: Keep file names descriptive and use standard [[WikiLinks]] format matching existing folder hierarchies.
