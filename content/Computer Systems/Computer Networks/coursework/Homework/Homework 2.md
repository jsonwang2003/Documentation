## 1.
Consider the following TCP congestion window graph, where the $x$ axis is in terms of round trips (RTTs), and the $y$ axis is in terms of segments

![[Pasted image 20261018143529.png]]

### a)
Label the $x$-axis showing RTT regions where the sender is operating in slowstart and AIMD congestion avoidance

**Answer**
![[Pasted image 20261019124121.png]]

### b) 
Annotate points on the graph where the following events occur:
1. Packet loss
2. Timeout
3. Fast retransmit / fast recovery

**Answer**
![[Pasted image 20261019140747.png]]

### c)
What is the value of `ssthresh` at each of the following points in time (measured in RTTs, counting from $0$)
1. RTT 5
2. RTT 14

**Answer**
`ssthresh(RTT = 5)` = 2
`ssthresh(RTT = 14)` = 3

---
## 2.
Consider the following network, where A-F are LANs assigned the indicated address, and R1-R3 are L3 switches. Each switch is connected to four networks using Ethernet NICs (named eth0-eth3) which are assigned IP addresses as shown in the diagram. You can assume the rest of the internet is accessible via R1's eth3 interface.

![[Pasted image 20261018143955.png]]

### a)
Below is a partial copy of R2's FIB. Fill in the rest using the fewest entries possible

| Destination                                  | Next Hop     | Interface |
| -------------------------------------------- | ------------ | --------- |
| 127.0.0.1/32                                 | 127.0.0.1    | lo0       |
| default                                      | 10.0.2.1     | eth3      |
| 192.168.16/21                                | 192.168.16.1 | eth0      |
| 10.0.2/8                                     | 10.0.2.2     | eth3      |
| 10.0.3/24                                    | 10.0.3.1     | eth2      |
| 192.168.8/22                                 | 192.168.8.1  | eth1      |
| 192.168.128/17                               | 10.0.2.2     | eth3      |
| 192.168.4/23<br>192.168.6/24<br>192.168.7/24 | 10.0.3.1     | eth2      |

### b) 
Suppose I am using a host with IP address 192.168.5.17. To which network is my host connected?

**Answer**
network E

$$
\begin{align*}
	192.168.5.17 \implies &11000000\ 10101000\ 00000101\ 00010001\\
	A = &11000000\ 10101000\ 1|0000000 \implies\text{not a match}\\
	B = &11000000\ 10101000\ 00010|000 \implies \text{not a match}\\
	C = &11000000\ 10101000\ 000010|00 \implies \text{not a match}\\
	D = &11000000\ 10101000\ 00000111| \implies \text{not a match}\\
	E = &11000000\ 10101000\ 0000010|0 \implies \text{match}\\
	F = &11000000\ 10101000\ 00000110| \implies \text{not a match}
\end{align*}
$$
### c)
Imagine a host on network A wanted to send a packet to a host on network F. What is the shortest path it could possibly take? Specifically, which routers would need to forward the packet?

**Answer**
![[Pasted image 20261019205155.png]]

---
## 3.
Consider the IP address 132.239.15.87

### a)
Suppose that we were still using class-based addressing. What type of network would this IP address be a part of?

**Answer**
$$
	132.239.15.87 \implies 10000100\ 11101111\ 00001111\ 01010111
$$

And since the start of the address is $10$, it implies the address is type B network
### b)
If the network administrator had decided to break the network in part a) into 64 different subnets, what would the subnet mask of the subnet to which this IP address belongs be?

**Answer**
With the need of $64$ subnets, we require $\log_{2}(64) = 6$ bits to represent the different subnets.

As such the subnet mask has $2 + 14 + 6 = 22$ prefix $1$'s followed by $32-22 = 10$ postfix $0$'s
### c)
What is the subnet number (address) of the subnet to which this IP address would be attached?

**Answer**
$$
	\begin{align*}
	&10000100\ 11101111\ 00001111\ 01010111\\
	\&\ &11111111\ 11111111\ 11111100\ 00000000\\
	\\
	=\  &10000100\ 11101111\ 00001100\ 00000000
	\end{align*}
$$

The first two 8-bit represents the network, which is not included in the subnet. After removing the first two bytes, the prefixed $6$ bits represents the subnet of this particular address:

$$
	000011_{2} \implies 3_{10}
$$
Therefore this address points towards subnet $3$
### d) 
Now suppose that we are using CIDR addressing instead of class-based addressing and subnets. What would the length of the CIDR prefix be for the physical network in part c) to which the host is attached?

**Answer**
The length of the CIDR prefix would be 22 as it matches the leading $1$'s in the netmask
### e)
What would the broadcast address for this network (i.e. the network from parts c) and d)) be?

**Answer**
Broadcast network is where the host bits all set to 1

$$
	\begin{align*}
	10000100\ 11101111\ 000011/11\ 11111111 = 132.239.15.255
	\end{align*}
$$

---
## 4.
Suppose $8$ flows share a $1$-Mbps link. Assume flows achieve throughputs of $4$ Kbps, $12$ Kbps, $16$ Kbps, $32$ Kbps, $64$ Kbps, $128$ Kbps, $256$ Kbps, $512$ Kbps

### a) 
What is Jain's Fairness index for this set of flows?

**Answer**
$$
\begin{align*}
	f(x_1,x_{2},\dots,x_{8}) &= \frac{\left( \sum_{i=1}^{8}x_{i} \right)^{2}}{n\sum_{i=1}^{n}x_{i}^{2}}\\
	&= \frac{(4+12+16+32+64+128+256+512)^{2}}{8(4^{2}+12^{2}+16^2+32^2+64^2+128^2+256^2+512^2)}\\
	&= \frac{1024^{2}}{8(16+144+256+1024+4096+16384+65536+262144)}\\
	&= \frac{1048576}{2796800}\\
	&=  0.37491990846682 \approx \boxed{0.375}
\end{align*}
$$
### b) 
What would the throughput of each flow need to be to achieve a fairness index of $1$?

**Answer**
$$
	\text{Throughput} = \frac{1 \text{ Mbps}}{\text{number of flows}} = \frac{1024 \text{ Kbps}}{8} = \boxed{128 \text{ Kbps}}
$$
### c)
Suppose a new flow arrived that caused congestion and led each of the senders to halve their sending rate according to AIMD. Assuming this new flow has a throughput of $64$ Kbps (after halving), what is the Fairness Index of the new set of rates?

**Answer**
New flow throughputs: $[2, 6, 8, 16, 32, 64, 64, 128, 256]$

$$
	\begin{align*}
	f(x_{1}, x_{2}, \dots, x_{9}) &= \frac{\left( \sum_{i=1}^{9}x_{i} \right)^{2}}{9\sum_{i=1}^{9}x_{i}^{2}}\\
	&= \frac{(2 + 6 + 8 + 16+32+64+64+128+256)^{2}}{9(2^{2}+6^2+8^2+16^2+32^2+64^2+64^2+128^2+256^2)}\\
	&= \frac{576^{2}}{9(4+36+64+256+3262+4096+4096+16384+65536)}\\
	&= \frac{331776}{9(93734)}\\
	&= \frac{331776}{843606}\\
	&= 0.3932831203192 \approx \boxed{0.393}
	\end{align*}
$$