## CPU Control and Datapath Execute Instruction Set
![[Pasted image 20260831230420.png]]

Control takes **program** as input, it interprets each instruction and tell the **Datapath** to operate on data via **ALU, memory** and **registers**

## CPU Components ― Single Cycle Execution
![[Pasted image 20260831231335.png]]

### Assumptions
- Every machine language instruction happens in $1$ clock cycle
- MIPS architecture (Microprocessor without interlocked pipeline stages)
- reg-reg architecture: all operands must be in registers (total 24)
- 3 Instruction Types: 
	1. R-Type: all data in registers (mostly arithmetic and logical operations)
	2. I-type: branches, memory transfers, constants
	3. J-type: jumps and calls

---
## R-type Instructions: Register to Register ALU ops
![[Pasted image 20260831231724.png]]

- `OPCODE`: tells operation to be performed
- `RS`: Source Register 1 (attached to "read register 1" input)
- `RT`: Source Register 2 (attached to "read register 2" input)
- `RD`: Destination Register (attached to "write register" input)
- `shamt`: Shift Amount for shift operations
- `FUNCT`: Tells specific variant of operation (eg. add/sub have same opcode)

### Steps for R-Type Instruction Operations
#### Step 1. Fetch instruction and advance PC
![[Pasted image 20260831232429.png]]
#### Step 2. Read two registers and set control signals
![[Pasted image 20260831232949.png]]
#### Step 3. Perform the ALU operation
![[Pasted image 20260831233551.png]]
#### Step 4: Write result to register
![[Pasted image 20260831233723.png]]

## I-Type: Store Instruction

![[Pasted image 20260831234352.png]]

- `OPCODE`: Tells operation to be performed
- `RS`: Base Address Register (attached to "read regester 1" input)
- `RT`: Source register whose value will be stored to memory (attached to "read register 2" input)
- `OFFSET`: Constant offset (added to the base address in `RS`)
### Steps for I-Type (Store) Instruction Operations
#### Step 1: Fetch instruction and advance PC
![[Pasted image 20260831232429.png]]
#### Step 2 (store): Read register values and set control signals
![[Pasted image 20260901143659.png]]
#### Step 3 (store): Compute the Address
![[Pasted image 20260831235426.png]]

#### Step 4 (store): Write the value to memory
![[Pasted image 20260901145348.png]]

## I-Type: Conditional Branch
![[Pasted image 20260901145442.png]]

- `RS`: Source Register 1 (attached to "read register 1" input)
- `RT`: Source Register 2 (attached to "read register 2" input)
- `BRANCH TARGET'S OFFSET`: Word offset, which we multiply by 4 (via << 2) to get Bit Offset, then add to PC+4 to get the address of the instruction to which we branch if `RS == RT` (PC-relative address)
 
### Steps for I-Type (Conditional) Instruction Operations
#### Step 1 (beq): Fetch instruction and advance PC 
![[Pasted image 20260831232429.png]]

#### Step 2 (beq): Read register values and set control signals
![[Pasted image 20260901143659.png]]

#### Step 3 (beq): Compare registers, calculate branch target, and choose new PC
![[Pasted image 20260901151519.png]]

## J-Type: Unconditional Branch
![[Pasted image 20260901150118.png]]

- `BRANCH TARGET ADDRESS`: Actual Address (in words) which we multiply by 4 (<< 2) to get 28-bit address, then concatenate to upper 4 bits of PC+4 to get the 32 bit address of instruction to which we branch unconditionally

![[Pasted image 20260901152457.png]]