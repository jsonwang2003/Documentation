---
title: "Latches & Flip-Flops"
description: "Fundamental bistable sequential storage elements: SR Latches, Level-Sensitive Gated Latches, D Latches, Master-Slave D Flip-Flops, Edge-Triggered timing, Enabled/Resetable FFs, transistor counts, and Latch vs. Flip-Flop comparisons."
aliases:
  - Latches & Flip-Flops
  - Latches and Flip-Flops
  - SR Latch
  - D Latch
  - D Flip-Flop
  - Sequential Storage
tags:
  - computer-systems
  - digital-systems
  - sequential-logic
  - latches
  - flip-flops
---
> [!abstract] Abstract
> **Latches and Flip-Flops** are bistable memory elements that form the foundation of sequential digital circuits. Unlike combinational circuits whose outputs depend solely on current inputs, sequential circuits utilize feedback loops to maintain internal state. A **Latch** is **level-sensitive**, continuously updating its output while its control/clock signal is active (transparent mode). A **Flip-Flop** is **edge-triggered**, sampling its input only during a specific clock transition (rising or falling edge) using a **Master-Slave** design. These primitives evolve from basic **SR Latches** to glitch-free **D Flip-Flops** with enable, reset, and preset capabilities.

---

## 1. SR Latch (Set/Reset Latch)

An **SR Latch** is the simplest asynchronous bistable circuit built from two cross-coupled NOR or NAND gates.

![[Pasted image 20260825003158.png]]
### Practical Application: Flight Attendant Call Button
* **Press Call ($S = 1$):** Light turns ON ($Q = 1$) and stays ON even after the button is released.
* **Press Cancel ($R = 1$):** Light turns OFF ($Q = 0$).

![[Pasted image 20260824172145.png]]
*Flight Attendant Call Button System*

![[Pasted image 20260824172313.png]]
*Cross-Coupled NOR SR Latch Schematic*

### Operational Analysis & States

| Inputs ($S, R$) | State | Output ($Q$) | Complement ($\overline{Q}$) | Description |
|:---:|:---:|:---:|:---:|---|
| $S = 1, R = 0$ | **Set** | $1$ | $0$ | Forces output $Q$ to $1$. |
| $S = 0, R = 1$ | **Reset** | $0$ | $1$ | Forces output $Q$ to $0$. |
| $S = 0, R = 0$ | **Hold / Memory** | $Q_{\text{prev}}$ | $\overline{Q}_{\text{prev}}$ | Retains previously stored state. |
| $S = 1, R = 1$ | **Invalid / Forbidden** | $0$ | $0$ | **Illegal State:** Violates $\overline{Q} = \text{NOT}(Q)$. |

![[Pasted image 20260824172733.png]]
*Set State Operation ($S=1, R=0 \implies Q=1$)*

![[Pasted image 20260824172830.png]]
*Reset State Operation ($S=0, R=1 \implies Q=0$)*

![[Pasted image 20260824172936.png]]
*Hold Memory State Operation ($S=0, R=0 \implies Q=Q_{\text{prev}}$)*

![[Pasted image 20260824173056.png]]
*Invalid Forbidden State ($S=1, R=1 \implies Q=0, \overline{Q}=0$)*

> [!danger] Hazard of the Invalid State ($S=1, R=1$)
> If $S$ and $R$ are simultaneously active ($1$) and then simultaneously released to $0$, the final state of $Q$ becomes **unpredictable** due to non-deterministic propagation delays along asymmetric feedback paths, causing output oscillation or settling into an unintended state (metastability).

### Characteristic Equation & State Diagram

Breaking the feedback loop ($Q(t) \to Q(t+\Delta)$) yields the next-state truth table and K-Map derivation:

| $S$ | $R$ | $Q(t)$ | $Q(t+\Delta)$ | Function |
|:---:|:---:|:---:|:---:|---|
| $0$ | $0$ | $0$ | $0$ | Hold ($Q_{\text{prev}}$) |
| $0$ | $0$ | $1$ | $1$ | Hold ($Q_{\text{prev}}$) |
| $0$ | $1$ | $0$ | $0$ | Reset ($0$) |
| $0$ | $1$ | $1$ | $0$ | Reset ($0$) |
| $1$ | $0$ | $0$ | $1$ | Set ($1$) |
| $1$ | $0$ | $1$ | $1$ | Set ($1$) |
| $1$ | $1$ | $0$ | $X$ | Forbidden / Disallowed |
| $1$ | $1$ | $1$ | $X$ | Forbidden / Disallowed |

![[Pasted image 20260824174000.png]]
*Feedback Path Interruption for Characteristic Derivation*

![[Pasted image 20260824174516.png]]
*K-Map for Next-State $Q(t+\Delta)$*

#### Characteristic Equation

$$Q(t+\Delta) = S + R'Q(t) \quad \text{subject to constraint: } S \cdot R = 0$$

![[Pasted image 20260824174635.png]]
*SR Latch State Transition Diagram*

---

## 2. Level-Sensitive Gated SR Latch & Clocking

Adding a control/enable signal ($C$ or $CLK$) prevents inputs from changing state uncontrollably.

![[Pasted image 20260824174827.png]]
*Gated / Level-Sensitive SR Latch*

* **When $C = 0$:** $S_{internal} = 0$ and $R_{internal} = 0$. The latch is forced into **Hold Mode** regardless of $S$ and $R$ inputs.
* **When $C = 1$:** External $S$ and $R$ inputs pass through to drive the internal SR latch.

> [!tip] Risk Reduction
> Adding an enable control signal reduces the probability of an invalid $S=1, R=1$ condition occurring by restricting state updates strictly to times when inputs are known to be stable.

### Clock Signal Metrics

Sequential logic circuits use periodic clock signals to synchronize data transfers across latches and flip-flops.

![[Pasted image 20260824175157.png]]
*Periodic Clock Waveform and Parameter Definitions*

* **Clock Period ($T$):** Total duration of one complete clock cycle (e.g., $T = 20\text{ ns}$).
* **Clock Frequency ($f$):** Number of cycles per second:
  $$f = \frac{1}{T} = \frac{1}{20\text{ ns}} = 50\text{ MHz}$$
* **Duty Cycle:** The ratio of time the clock is HIGH during one period:
  $$\text{Duty Cycle} = \frac{T_{HIGH}}{T_{TOTAL}} \times 100\% \quad (50\% \text{ typical})$$

---

## 3. Level-Sensitive D Latch

The **D Latch (Data Latch)** eliminates the $S=1, R=1$ invalid state by placing an inverter between the $S$ and $R$ inputs.

![[Pasted image 20260824234831.png]]
*D Latch Block Symbol*

![[Pasted image 20260824175633.png]]
*Gated D Latch Circuit Schematic using an Inverter*

### Operational Summary

* **When $CLK = 1$ (Transparent Mode):** Inputs set $S = D$ and $R = \overline{D}$. The input $D$ passes directly to output $Q$.
* **When $CLK = 0$ (Opaque / Hold Mode):** Internal signals drop to $S = 0, R = 0$. Output $Q$ holds its previous value.

### Transistor-Level Complexity Breakdown

Constructing a standard gated D latch from CMOS gates requires **22 transistors ($22T$)**:

$$\begin{aligned}
\text{Total Transistors} &= (2 \times \text{AND}) + (1 \times \text{NOT}) + (1 \times \text{Cross-Coupled NOR Latch}) \\
&= (2 \times 6T) + (1 \times 2T) + (2 \times 4T) \\
&= 12T + 2T + 8T = \mathbf{22T}
\end{aligned}$$

![[Pasted image 20260824233933.png]]
*Transistor-Level Diagram of Static CMOS D Latch*

### D Latch Truth Table

| $CLK$ | $D$ | $\overline{D}$ | Internal $S$ | Internal $R$ | $Q$ | $\overline{Q}$ | Operational State |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| $0$ | $X$ | $\overline{X}$ | $0$ | $0$ | $Q_{\text{prev}}$ | $\overline{Q}_{\text{prev}}$ | **Hold (Opaque)** |
| $1$ | $0$ | $1$ | $0$ | $1$ | $0$ | $1$ | **Reset ($0$)** |
| $1$ | $1$ | $0$ | $1$ | $0$ | $1$ | $0$ | **Set ($1$)** |

---

## 4. D Flip-Flop (Master-Slave Edge-Triggered Storage)

Level-sensitive latches remain transparent as long as $CLK = 1$, allowing data to propagate through multiple cascaded latches within a single clock cycle (**race condition**). A **D Flip-Flop** solves this by sampling input $D$ strictly on a **clock transition (edge)**.

![[Pasted image 20260824235634.png]]
*Edge-Triggered D Flip-Flop Symbol*

### Master-Slave Architecture

A Master-Slave D Flip-Flop connects two D latches in series driven by inverted clock signals:

![[Pasted image 20260824234925.png]]
*Master-Slave D Flip-Flop Internal Architecture*

1. **Phase 1 ($CLK = 0$):**
   * **Master Latch ($C_m = 1$):** Transparent. $D$ is sampled into internal node $Q_m$.
   * **Slave Latch ($C_s = 0$):** Opaque (Hold). Output $Q$ maintains previous state.
2. **Phase 2 ($CLK \to 1$ Transition / Rising Edge):**
   * **Master Latch ($C_m = 0$):** Becomes opaque, isolating $Q_m$ from input changes.
   * **Slave Latch ($C_s = 1$):** Becomes transparent, passing captured node $Q_m$ directly to final output $Q$.

#### Characteristic Equation
$$Q(t+1) = D(t) \quad \text{sampled at the active clock edge}$$

### Rising Edge vs. Falling Edge Comparison

| Feature | Rising Edge D Flip-Flop | Falling Edge D Flip-Flop |
|---|:---:|:---:|
| **Symbol** | ![[Pasted image 20260825000908.png]] | ![[Pasted image 20260825000929.png]] |
| **Inverter Placement** | Inverter drives Master Clock ($C_m$) | Inverter drives Slave Clock ($C_s$) |
| **Active Edge** | $0 \to 1$ Transition | $1 \to 0$ Transition |
| **Timing Waveform** | ![[Pasted image 20260825001116.png]] | ![[Pasted image 20260825001123.png]] |

---

## 5. Enhanced D Flip-Flop Features

### Enabled D Flip-Flop (Load Enable $EN$ / $LD$)

Instead of gating the clock line (which introduces clock skew), an **Enabled D Flip-Flop** uses an input multiplexer controlled by $EN$ to loop the output $Q$ back when disabled:

![[Pasted image 20260825001350.png]]
*Enabled D Flip-Flop using Feedback Multiplexer*

* **When $EN = 1$:** Multiplexer selects new input $D$. $Q$ updates on active clock edge.
* **When $EN = 0$:** Multiplexer selects current $Q$. $Q$ retains state across cycles.

$$D_{\text{new}} = EN' \cdot Q + EN \cdot D_{\text{in}}$$

### Reset and Preset Controls
Flip-flops include controls to force the initial state to $0$ (Reset $R$) or $1$ (Set/Preset $S$).

#### Reset (set state to 0) ― $R$
- synchronous: $D_{\text{new}} = \bar{R} \cdot D_{\text{old}}$ (when next clock edge arrives)
- asynchronous: doesn't wait for clock (inside FF)

![[Pasted image 20260825001546.png]]

#### Preset or set (set state to 1) ― S (sometimes P)
- synchronous: $D_{\text{new}} = D_{\text{old}} + S$ (when next clock edge arrives)
- asynchronous: doesn't wait for clock (inside FF)

![[Pasted image 20260825001735.png]]

#### Both reset and preset
- $D_{\text{new}} = \bar{R} \cdot D_{\text{old}} + S \qquad \text{(set-dominant)} S=1, R \text{ doesn't matter}$
- $D_{\text{new}} = \bar{R} \cdot D_{\text{old}} + \bar{R}S \qquad \text{(reset-dominant)} R=1, \bar{R} = 0 \text{ dominate}$

![[Pasted image 20260825002115.png]]

#### Synchronous Control
Updates occur **only on the active clock edge**. The control signal is gated into the data input $D_{new}$:

$$\begin{aligned}
\text{Synchronous Reset Only: } D_{\text{new}} &= R' \cdot D_{\text{old}} \\
\text{Synchronous Preset Only: } D_{\text{new}} &= D_{\text{old}} + S
\end{aligned}$$

#### Asynchronous Control
Overrides the clock line and updates state **immediately** via direct clear/preset lines in the internal latches.

#### Priority Dominance Equations

$$\begin{aligned}
\text{Set-Dominant Logic: } D_{\text{new}} &= R' \cdot D_{\text{old}} + S \quad &&(\text{If } S=1 \implies D_{\text{new}}=1) \\
\text{Reset-Dominant Logic: } D_{\text{new}} &= R' \cdot (D_{\text{old}} + S) \quad &&(\text{If } R=1 \implies D_{\text{new}}=0)
\end{aligned}$$

---

## 6. Evolution of Bit Storage Elements

| Memory Element | Key Feature | Primary Disadvantage |
|---|---|---|
| **SR Latch**<br>![[Pasted image 20260824235944.png]] | Cross-coupled NOR feedback. Simple $S=1$ set, $R=1$ reset. | $S=1, R=1$ creates invalid/forbidden output state. |
| **Gated SR Latch**<br>![[Pasted image 20260825000057.png]] | Added enable signal $C$. Limits update window. | Must guarantee $S=1, R=1$ never occurs while $C=1$. |
| **D Latch**<br>![[Pasted image 20260825000259.png]] | Single data input $D$. Eliminates $S=1, R=1$ state. | **Level-sensitive:** Remains transparent as long as $C=1$. |
| **D Flip-Flop**<br>![[Pasted image 20260825000447.png]] | **Edge-triggered Master-Slave design.** Samples only on edge. | Larger internal gate count than basic latches. |

---

## 7. Latch vs. Flip-Flop Waveform Comparison

The timing diagram below illustrates the fundamental operational difference between a **Level-Sensitive D Latch** and a **Positive Edge-Triggered D Flip-Flop**:

![[Pasted image 20260825002525.png]]
*Timing Comparison between D Latch and Positive Edge-Triggered D Flip-Flop*

* **D Latch Output ($Q_{\text{Latch}}$):** Continuously tracks input $D$ whenever $CLK = 1$ (**transparent window**).
* **D Flip-Flop Output ($Q_{\text{FF}}$):** Captures the value of $D$ **only at the exact moment** $CLK$ transitions from $0 \to 1$ and holds it stable for the rest of the cycle.

---

## Related Notes

- [[Computer Systems/Digital Systems/ALU/Mux & Demux|Mux & Demux]]
- [[Computer Systems/Digital Systems/ALU/Arithmetic Logic Unit|Arithmetic Logic Unit Integration]]
- [[Computer Systems/Digital Systems/Logic Design/index|Logic Design & K-Maps]]
- [[Computer Systems/Digital Systems/index|Digital Systems Index]]