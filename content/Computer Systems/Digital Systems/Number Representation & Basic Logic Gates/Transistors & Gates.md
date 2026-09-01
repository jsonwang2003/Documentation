---
title: "Transistors & Gates"
description: "Electrical foundations of digital circuits, CMOS switches (nMOS/pMOS), Pull-Up/Pull-Down networks, transmission gates, and transistor-level implementations of universal NAND/NOR logic gates."
aliases:
  - Transistors & Gates
  - CMOS Logic
  - Transistors
  - CMOS
tags:
  - computer-systems
  - digital-systems
  - cmos
  - transistors
  - logic-gates
---
> [!abstract] Abstract
> **Electronic switches** serve as the physical foundation of modern binary digital systems. **CMOS (Complementary Metal-Oxide-Semiconductor)** technology combines **nMOS** transistors (which form Pull-Down Networks connected to ground) and **pMOS** transistors (which form Pull-Up Networks connected to supply voltage $V_{DD}$). By combining nMOS and pMOS in parallel, **Transmission Gates** act as ideal bilateral switches that avoid voltage degradation. Transistor-level implementations of **NAND** and **NOR** gates demonstrate functional universality, enabling all Boolean logic functions to be synthesized from basic switch networks.

---

## Electrical Terminology & Physical Foundations

Digital logic signals are physical voltage levels. Analyzing gate delays and dynamic power consumption requires understanding key electrical parameters:

* **Voltage ($V$):** The difference in electrical potential between two points, measured in Volts. Represents digital logic states ($V_{DD}$ for Logic 1, Ground/GND for Logic 0).
* **Current ($I$):** The flow rate of charged particles through a conductor, measured in Amperes.
* **Resistance ($R$):** The tendency of a material or wire to impede current flow, measured in Ohms ($\Omega$).
  
  $$\text{Ohm's Law: } V = IR$$

* **Capacitance ($C$):** The ratio of electric charge change to electric potential change, measured in Farads ($F$). Gate terminals and interconnect wires behave as parasitic capacitors.

  $$I = C \cdot \frac{dQ}{dt}$$

  $$\Delta V = I \cdot \frac{\Delta t}{C}$$

> [!info] Physical Switching Behavior
> Charging a capacitance $C$ to voltage $\Delta V$ requires finite time $\Delta t$. This charging/discharging current limits the maximum operating clock frequency of digital circuits.

---

## CMOS Switch Fundamentals

**CMOS (Complementary Metal-Oxide-Semiconductor)** technology uses paired n-channel (nMOS) and p-channel (pMOS) field-effect transistors to construct digital logic with near-zero static power dissipation.

![[Pasted image 20260806175955.png]]
*CMOS Complementary Transistor Pair Structure*

### nMOS vs. pMOS Transistors

![[Pasted image 20260806180259.png]]
*nMOS Transistor Switch*

![[Pasted image 20260806180321.png]]
*pMOS Transistor Switch*

| Property                       | nMOS Transistor                         | pMOS Transistor                          |
| ------------------------------ | --------------------------------------- | ---------------------------------------- |
| **Active Control Input**       | Logic 1 ($V_{DD}$ / positive charge)    | Logic 0 ($GND$ / negative charge)        |
| **Switch State when Gate = 1** | **ON** (Closed Circuit)                 | **OFF** (Open Circuit)                   |
| **Switch State when Gate = 0** | **OFF** (Open Circuit)                  | **ON** (Closed Circuit)                  |
| **Strong Signal Transmission** | Passes **0** well (Strong $GND$)        | Passes **1** well (Strong $V_{DD}$)      |
| **Weak Signal Transmission**   | Passes **1** poorly ($V_{DD} - V_{th}$) | Passes **0** poorly ($GND + \|V_{th}\|$) |
| **Network Configuration**      | **Pull-Down Network (PDN)**             | **Pull-Up Network (PUN)**                |
| **Rail Connection**            | Connected to Ground ($GND$)             | Connected to Power ($V_{DD}$)            |

---

## Transistor Circuit Design: PUN and PDN

A static CMOS gate consists of two complementary networks operating mutually exclusively:

![[Pasted image 20260806180630.png]]
*Static CMOS Dual Network Structure (Pull-Up and Pull-Down Networks)*

1. **Pull-Up Network (PUN):** Composed exclusively of **pMOS** transistors connected between $V_{DD}$ and the output node. When active, it pulls the output to a strong Logic 1.
2. **Pull-Down Network (PDN):** Composed exclusively of **nMOS** transistors connected between the output node and $GND$. When active, it pulls the output to a strong Logic 0.

> [!tip] Dual Network Rule
> For any valid input combination, **either** the PUN connects the output to $V_{DD}$ **or** the PDN connects the output to $GND$. They are never ON simultaneously (which would cause a short circuit) nor both OFF simultaneously (which would leave the output floating).

---

## Transmission (Pass) Gates

Using a single transistor type as a switch introduces signal degradation:
* An nMOS switch alone passes $0$s strongly, but degrades $1$s.
* A pMOS switch alone passes $1$s strongly, but degrades $0$s.

A **Transmission Gate (TG)** combines an nMOS and a pMOS transistor in parallel to create a full-rail bilateral switch.

![[Pasted image 20260806181336.png]]
*Transmission Gate Circuit Schematic and Symbol*

### Operational States

* **Enabled ($EN = 1, EN' = 0$):** Both transistors turn **ON**. Input $A$ connects directly to output $B$. The nMOS strongly passes low voltage ($0$), while the pMOS strongly passes high voltage ($1$).
* **Disabled ($EN = 0, EN' = 1$):** Both transistors turn **OFF**. Input $A$ is disconnected from $B$. The output node enters a High-Impedance state (**$Z$** / sleep mode).

| Enable ($EN$) | Complement ($EN'$) | nMOS State | pMOS State | Output ($B$) |
|:---:|:---:|:---:|:---:|:---:|
| $0$ | $1$ | OFF | OFF | High-Impedance (**$Z$**) |
| $1$ | $0$ | ON | ON | Connected to Input ($A$) |

---

## Transistor-Level Gate Implementations & Universality

Any Boolean function can be constructed using only **NAND** gates or only **NOR** gates (Functional Universality).

---

### Universal Logic: NAND Implementations

A basic 2-input NAND gate consists of **4 transistors**: 2 pMOS in parallel (PUN) and 2 nMOS in series (PDN).

#### 1. NOT Gate (Inverter)
Constructed by wiring a 2-input NAND gate with inputs tied together (or using a standard 2-transistor CMOS inverter):

![[Pasted image 20260806182141.png]]
*NAND-configured NOT Gate (4 Transistors)*

* **Transistor Count:** **4 Transistors** (when using a 2-input NAND) / **2 Transistors** (standard CMOS Inverter).

#### 2. AND Gate
Constructed by cascading a 2-input NAND gate with an Inverter:

![[Pasted image 20260806182537.png]]
*NAND-based AND Gate*

* **Transistor Count:** **6 Transistors** ($4 \text{ [NAND]} + 2 \text{ [Inverter]}$).

#### 3. OR Gate
Constructed using De Morgan's Law ($\overline{A \cdot B} = \overline{A} + \overline{B} \implies A + B = \overline{\overline{A} \cdot \overline{B}}$) by inverting inputs before feeding a NAND gate:

![[Pasted image 20260806183035.png]]
*NAND-based OR Gate Structure*

* **Transistor Count:** **8 Transistors** ($2 \times 2 \text{ [Inverters]} + 4 \text{ [NAND]}$).

---

### Universal Logic: NOR Implementations

A basic 2-input NOR gate consists of **4 transistors**: 2 pMOS in series (PUN) and 2 nMOS in parallel (PDN).

#### 1. NOT Gate (Inverter)
Constructed by tying the inputs of a 2-input NOR gate together:

![[Pasted image 20260806200919.png]]
*NOR-configured NOT Gate (4 Transistors)*

* **Transistor Count:** **4 Transistors** (when using a 2-input NOR).

#### 2. OR Gate
Constructed by cascading a 2-input NOR gate with an Inverter:

![[Pasted image 20260806201250.png]]
*NOR-based OR Gate*

* **Transistor Count:** **6 Transistors** ($4 \text{ [NOR]} + 2 \text{ [Inverter]}$).

#### 3. AND Gate
Constructed using De Morgan's Law ($\overline{A + B} = \overline{A} \cdot \overline{B} \implies A \cdot B = \overline{\overline{A} + \overline{B}}$) by inverting inputs before feeding a NOR gate:

![[Pasted image 20260806200632.png]]
*NOR-based AND Gate Structure*

* **Transistor Count:** **8 Transistors** ($2 \times 2 \text{ [Inverters]} + 4 \text{ [NOR]}$).

---

## Summary of Gate Transistor Counts

| Desired Logic Gate | NAND-Based Construction | NOR-Based Construction | Total Transistor Count |
|---|---|---|:---:|
| **NOT** | NAND with inputs tied | NOR with inputs tied | **4** (or 2 for standalone Inverter) |
| **AND** | NAND + NOT | NOTs + NOR | **6** (via NAND) / **8** (via NOR) |
| **OR** | NOTs + NAND | NOR + NOT | **8** (via NAND) / **6** (via NOR) |

---

## Related Notes

- [[Number Systems and Boolean Algebra|Number Systems and Boolean Algebra]]
- [[Computer Systems/Digital Systems/Combinational Logic Design|Combinational Logic Design]]
- [[Computer Systems/Digital Systems/SOP, POS, K-Maps & Logic Simplification|SOP, POS, K-Maps & Logic Simplification]]
- [[Computer Systems/Digital Systems/index|Digital Systems Index]]