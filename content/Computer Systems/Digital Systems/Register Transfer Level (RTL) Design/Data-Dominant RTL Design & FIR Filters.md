---
description: "Comparison of Control-Dominant vs. Data-Dominant digital architectures, step-by-step RTL synthesis of a Finite Impulse Response (FIR) Filter, and hardware parallelism vs. software execution performance analysis."
aliases:
  - Data-Dominant RTL Design
  - FIR Filter Design
  - Data-Dominant Architecture
  - FIR Filter Hardware
tags:
  - computer-systems
  - digital-systems
  - rtl-design
  - fir-filter
  - datapath
---
> [!abstract] Abstract
> Digital systems are broadly categorized as either **Control-Dominant** or **Data-Dominant**. Control-dominant systems prioritize complex decision logic with simple datapaths (e.g., traffic light controllers), whereas **data-dominant systems** feature extensive, highly parallel datapaths managed by minimal control logic (e.g., Digital Signal Processors, FIR Filters). Implementing a **Finite Impulse Response (FIR) Filter** in custom hardware demonstrates how data-dominant RTL designs achieve orders-of-magnitude performance speedups over software by processing wide array operations in parallel along logarithmic adder trees.

---

## 1. Architectural Classification: Control-Dominant vs. Data-Dominant

RTL designs balance control complexity against datapath computation density depending on the target application:

| Attribute | Control-Dominant Design | Data-Dominant Design |
|---|---|---|
| **Controller Complexity** | **High:** Complex state transition diagrams, many branching conditions. | **Low:** Minimal state count, often simple iterative loops or single-state control. |
| **Datapath Complexity** | **Low:** Simple registers, basic gates, or minimal arithmetic operations. | **High:** Extensive register chains, parallel multipliers, adders, and accumulators. |
| **Primary Bottleneck** | Decision logic latency & state decoding overhead. | Data propagation delay through deep arithmetic operators (**Critical Path**). |
| **Typical Examples** | CPU Control Units, Protocol Handshakers, Traffic Controllers. | Digital Signal Processors (DSP), **FIR Filters**, Video/Audio Codecs, Neural Net Accelerators. |

---

## 2. Case Study: Finite Impulse Response (FIR) Filter

A **Finite Impulse Response (FIR) Filter** transforms an input digital stream $x(t)$ into a filtered output stream $y(t)$ by computing a configurable weighted sum of present and past input samples.

```
                      ┌──────┐
x(t) ───┬────────────►│ c_0  ├───────────┐
        │             └──────┘           │
      ┌─▼──┐          ┌──────┐         ┌─▼─┐
      │z⁻¹ │───┬─────►│ c_1  ├────────►│ + │
      └────┘   │      └──────┘         └─┬─┘
             ┌─▼──┐   ┌──────┐           │     ┌───┐
             │z⁻¹ │──►│ c_2  ├───────────┴────►│ + ├───► y(t)
             └────┘   └──────┘                 └───┘
```

### Mathematical Definition
For an $N$-tap FIR filter, the transfer function is expressed as:

$$y(t) = \sum_{i=0}^{N-1} c_i \cdot x(t-i)$$

#### 3-Tap FIR Filter Equation
$$y(t) = c_0 \cdot x(t) + c_1 \cdot x(t-1) + c_2 \cdot x(t-2)$$

* **Filter Taps ($N$):** The number of past input samples preserved in the delay chain.
* **Filter Coefficients ($c_i$):** User-configurable constants that define the frequency response (e.g., low-pass, high-pass, or band-pass filtering).
  * **Small $N$:** Minimal delay/filtering.
  * **Large $N$:** Strong noise attenuation, sharper cutoff frequencies, but higher hardware resource usage.

![[Pasted image 20260826170456.png]]
*Simple Moving Average Filtering Effect*

---

## 3. RTL Design Pipeline for a 3-Tap FIR Filter

### Step 1: Capture Behavior (HLSM)
Because the datapath processes samples continuously on every clock tick, the controller HLSM requires minimal control states.

![[Pasted image 20260827135717.png]]
*High-Level State Machine (HLSM) for FIR Filter Control*

### Step 2: Datapath Construction
To execute $y(t) = c_0 x(t) + c_1 x(t-1) + c_2 x(t-2)$, the datapath integrates:
1. **Sample Delay Chain:** A cascaded shift register chain ($x_{t0}, x_{t1}, x_{t2}$) storing incoming samples $x(t-i)$.
2. **Coefficient Registers:** Storage registers for constants $c_0, c_1, c_2$ (with write-enable logic for configuration).
3. **Parallel Multipliers:** Three array multipliers computing $c_i \cdot x(t-i)$ concurrently.
4. **Adder Tree:** Combinational adder stages summing the partial products into output $y(t)$.

![[Pasted image 20260827140021.png]]
*3-Tap FIR Filter Datapath Architecture*

### Step 3 & 4: Controller Connection & FSM Synthesis
The simple controller asserts load/clear lines to configure coefficients and manage sample propagation.

![[Pasted image 20260827140154.png]]
*Interfacing Datapath with Minimal Controller Block*

---
## 4. Hardware (RTL Circuit) vs. Software Performance Analysis

Evaluating a **100-tap FIR filter** demonstrates the profound performance advantage of hardware spatial parallelism over software temporal execution.

![[Pasted image 20260827140824.png]]
*Critical Path through a 100-Tap FIR Filter Hardware Adder Tree*

### Gate-Delay Model Assumptions
* **Adder Delay:** $2$ gate delays per addition.
* **Multiplier Delay:** $20$ gate delays per multiplication.
* **Software Instruction Delay:** $10$ gate delays per instruction execution cycle.

### Critical Path & Performance Breakdown

#### 1. Hardware RTL Implementation (Parallel Spatial Processing)
In hardware, all $100$ multiplications execute concurrently in parallel ($20$ gate delays). The resulting $100$ products are summed via a **balanced binary adder tree** of depth $\lceil \log_2(100) \rceil = 7$ adder stages.

$$\begin{aligned}
\text{Longest Critical Path} &= \text{Multiplier Delay} + \left( \lceil \log_2(N) \rceil \times \text{Adder Delay} \right) \\
&= 20 + (7 \times 2) \\
&= \mathbf{34 \text{ Gate Delays per Sample Output}}
\end{aligned}$$

#### 2. Software Implementation (Sequential Loop Processing)
A general-purpose processor must process the 100-tap filter sequentially inside a loop ($100$ multiplications + $100$ additions). Assuming $2$ instructions per operation:

$$\begin{aligned}
\text{Total Operations} &= (100 \text{ mults} \times 2 \text{ inst}) + (100 \text{ adds} \times 2 \text{ inst}) = 400 \text{ instructions} \\
\text{Total Latency} &= 400 \text{ instructions} \times 10 \text{ gate delays/instruction} \\
&= \mathbf{4000 \text{ Gate Delays per Sample Output}}
\end{aligned}$$

### Performance Summary Matrix

| Metric | Software Execution (CPU) | Dedicated RTL Circuit (Data-Dominant) |
|---|:---:|:---:|
| **Execution Paradigm** | Sequential Iterative Loop | Parallel Spatial Hardware Datapath |
| **3-Tap Latency** | $\approx 240$ Gate Delays | **$24$ Gate Delays** ($1 \text{ mult} + 2 \text{ adders}$) |
| **100-Tap Latency** | $4000$ Gate Delays | **$34$ Gate Delays** ($1 \text{ mult} + 7 \text{ tree adders}$) |
| **Speedup (100-Tap)** | $1.0\times$ (Baseline) | **$\approx 117.6\times$ Faster** |

> [!tip] Algorithmic Scaling Advantage
> As the number of filter taps $N$ grows, software latency increases **linearly** $\mathcal{O}(N)$, whereas hardware critical path latency grows **logarithmically** $\mathcal{O}(\log_2 N)$ thanks to parallel adder tree structures.

---

## Related Notes

- [[RTL Design Process]]
- [[High Level State Machines|High-Level State Machines]]
- [[Computer Systems/Digital Systems/Sequential Circuit/Timing Constraints in Sequential Designs|Timing Constraints in Sequential Designs]]
- [[Computer Systems/Digital Systems/ALU/Multiplier & Divider|Multiplier & Divider]]
- [[Computer Systems/Digital Systems/Sequential Circuit/index|Sequential Circuits Index]]