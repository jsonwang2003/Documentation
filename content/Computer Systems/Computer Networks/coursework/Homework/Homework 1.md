# 1.  
Suppose that you want to send a (short) Slack message using your computer that is connected to the Internet over an Ethernet network. Assume that Slack sends messages using HTTP requests. 
## A)  
Draw a diagram of an Ethernet frame carrying your message as it appears on the wire. In your diagram, show the location of  1) various protocol headers,  2) application-layer payload (message),  3) start of the frame, 4) end of the frame. It is OK if you don’t know the particular protocol in use at each layer, but you should indicate to which layer each header corresponds.

**Answer**
![[Pasted image 20260930120149.png]]

## B) 
For the Ethernet frame you drew in 1a), which is the first header that the receiving host will process? Which is the first header that your computer added?

**Answer**
First header receiver process: Ethernet Header
First header added: HTTP header

---
# 2.  
Consider a code on  four-bit strings that contains (only) the following four codewords:$0001, 1101, 0110, 1011$
## A) 
What is Hamming Distance of this code?

**Answer**
- $0001 \to 1101 = 2$
- $0001 \to 0110 = 3$
- $0001 \to 1011 = 2$
- $1101 \to 0110 = 3$
- $1101 \to 1011 = 2$
- $0110 \to 1011 = 3$

The minimum distance from any two code = 2, therefore the Hamming Distance of this code = 2

## B) 
What is the rate of this code if we use it to encode two-bit strings? Is it efficient? If it is not efficient, please explain.

**Answer**
$$
	\text{rate} = \frac{\text{number of code bits}}{\log_{2}(\text{number of possible combinations of bits sent})} = \frac{2}{4} = 0.5
$$
This code is not efficient as, from part a), the list of hamming distance between every unique pair of encodings are not uniform.

## C) 
How many bit flips can be detected using this code? How many bit flips can be corrected

**Answer**
$$
	\text{Hamming distance} = 2 = 2d+1 \implies d = \frac{1}{2}
$$
- Number of bit flips can be detected: $2d = 1$
- Number of bit flips can be corrected: $d = \frac{1}{2} \to 0$

---
# 3. 
Suppose you are transmitting a $16$-bit message using a protocol that protects the payload with a CRC defined by the polynomial $C(x) = x^{8} + x^{5}+ x^{4}+ 1$

## A) 
How many bits are actually required to transmit the message and the frame check sequence? How many bits correspond to the frame check sequence?

**Answer**
$C(k)$ has a polynomial of degree $k=8$
Therefore, the frame check sequence requires $8$ bits

The message is $16$-bits worth
Therefore, the total message requires $16+8 = 24$ bits
## B) 
Suppose that exactly one bit error occurred during transmission. Will the receiver be able to detect it? Why or why not?

**Answer**
The receiver is able to detect one bit error because of the property with any single bit error can be guaranteed to be detected if the $C(x)$ has a nonzero coefficient for $x^{k}$ term and $x^{0}$ term.

Suppose a single bit error occurred at index $i$. This can be represented as $x^{i}$ in polynomial notation. In order for $x^{i}$ to be divisible by $C(x)$, $C(x)$ must be equal to $x^{k}$ where $k\leq i$. With a nonzero coefficient for $x^{k}$ and $x^{0}$ guarantees there is more than 1 polynomial terms, which can not divide a single bit $x^{i}$ evenly.
## C) 
On the contrary, suppose the receiver received the following bits without error: `010101011010101011111111`. What was the payload?

**Answer**
Assume no error, then the payload will be the $16$ bits at the front, stripping the eight 1's on the back as they are the frame check sequence from the CRC encoding. Therefore, payload = `0101010110101010`

---
# 4. 
Consider a sender and receiver that implement a sliding window protocol using selective retransmission, as described in Section 12.4 of the textbook. Assume that: 
- The frame sequence number space ranges from 0 to 4 
- The send window and receive window size are 2. 
- The receiver sends a cumulative ACK each time it receives a frame from the sender 
- The one-way delay between sender and receiver is slightly less than 1ms 
- The sender sends frames no faster than once per millisecond 
- The sender uses a timeout of 4 ms.

![[Pasted image 20260930211512.png]]

## A) 
As you can see, the transmission of ACK 1 was lost. You should extend the diagram above until the sender receives the ACK for Frame 4 assuming no further frame (or ACK) losses.

![[Pasted image 20260930211543.png]]

Where LAR stands for Last Acknowledgement Received, LFS is Last Frame Sent, LFR is Last (in-order) Frame Received, and LAF is Largest Acceptable Frame.

**Answer**
![[Pasted image 20261002150139.png]]

## B)
Using your extended trace from the prior question, draw a similar state diagram for the Sender and Receiver at t=4.5ms. Include arrows for LAR, LFS, LFR, and LAF.

![[Pasted image 20261002172029.png]]

## C) 
And, finally, draw a similar diagram at the end of the trace from part (a) (i.e., the moment after the Sender receives the ACK for Packet 4 but before it sends anything further).

![[Pasted image 20261002172043.png]]