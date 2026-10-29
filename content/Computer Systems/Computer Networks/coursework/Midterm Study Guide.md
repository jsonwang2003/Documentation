## Document Overview & Learning Objectives

This study guide provides a comprehensive review of the concepts, protocols, algorithms, and mathematical frameworks presented in CSE 123: Computer Networks. Designed for rigorous exam preparation and concept mastery, this guide systematically covers the foundational abstractions of network architecture, physical link framing, error detection, transport reliability, congestion control, packet switching, IP addressing, CIDR subnetting, forwarding, ARP, middleboxes (NAT/IPv6), and application-layer services (DNS and DASH video streaming).

---
## Section 1: Foundations & Network Architecture

### 1.1 Fundamental Protocol Layering & Modularity

Computer networks are complex systems designed to connect heterogeneous hardware and software systems globally. To manage this systemic complexity, networks rely on modular protocol layering. Each layer in the network stack provides a specific set of services to the layer above it while hiding the implementation details of the layers below.

As established in the introductory lectures, the five-layer Internet protocol stack organizes network communication into distinct functional boundaries:

1. **Application Layer**: Supports network applications and user protocols (e.g., HTTP, DNS, DASH).
2. **Transport Layer**: Manages process-to-process data delivery, multiplexing, flow control, and reliability (e.g., TCP, UDP).
3. **Network Layer**: Handles host-to-host packet routing and logical addressing across interconnected networks (e.g., IP, ICMP).
4. **Link / MAC Layer**: Controls frame movement across a single physical network link or bridged local area network (e.g., Ethernet, 802.11 Wi-Fi, SONET).
5. **Physical Layer**: Transmits raw binary bit streams over physical media using electrical, optical, or radio signals.

#### Key Layering Concepts

- **Encapsulation**: As data moves down the protocol stack from the application layer to the physical link, each protocol adds its own control header (and occasionally a trailer) to the payload received from the higher layer. Upon reception, the peer stack performs decapsulation, stripping headers layer-by-layer.
- **Syntax vs. Semantics**: A protocol standard defines both syntax (the exact format, field offsets, and encoding of message packets) and semantics (the operational meaning of control fields and the required state machine actions taken upon message receipt).

### 1.2 Performance Metrics & Mathematical Models

Evaluating network performance requires precise quantification of link capacity, delay components, and data volume abstractions.

#### Link Performance Definitions

- **Bandwidth (****R****)**: The nominal physical bit-rate capacity of a communication link, measured in bits per second (bps, Mbps, Gbps).
- **Throughput**: The actual rate of successful data delivery over a channel, bounded by link bandwidth, packet overhead, queuing, and protocol mechanics.

#### Latency Components

Total end-to-end latency ($T_{\text{total}}$) for a packet traversing a network path is the sum of four distinct delay components:

$$
T_{\text{total}} = T_{\text{prop}} + T_{\text{trans}} + T_{\text{queue}} + T_{\text{proc}}
$$

- **Propagation Delay (****$T_{\text{prop}}$****)**: The time required for a physical signal to travel across the physical medium over distance $D$ at propagation speed $C$: $T_{\text{prop}} = \frac{D}{C}$ (Note: Propagation speed $C$ _in optical fiber or copper wire is typically_ $\approx 2 \times 10^8 \text{ m/s}$, or roughly two-thirds the speed of light in a vacuum).
- **Transmission Delay (****T_{\text{trans}}****)**: The time required to push all bits of a packet of size L bits onto the transmission link operating at rate R bps: T_{\text{trans}} = \frac{L}{R}
- **Queuing Delay (****T_{\text{queue}}****)**: The variable duration a packet spends waiting in router or switch buffer queues prior to transmission. T_{\text{queue}} depends directly on traffic burstiness and buffer arrival rates.
- **Processing Delay (****T_{\text{proc}}****)**: The static time required by a network node to inspect packet headers, verify checksums, and execute forwarding table lookups.

#### Bandwidth-Delay Product (BDP)

The Bandwidth-Delay Product represents the maximum volume of unacknowledged data that can exist "in flight" within the network pipe at any single moment. It is defined as the product of the link bandwidth ($R$) and the Round-Trip Time ($\text{RTT}$):

$$
\text{BDP} = R \times \text{RTT}
$$

A link with a high BDP (often termed a "fat pipe") requires large transport protocol window sizes to maintain full channel utilization.

#### Jitter & Statistical Multiplexing

- **Jitter**: The variation in packet delay over time. In real-time streaming applications, uneven inter-packet arrival gaps cause playback distortion, requiring client-side playback buffers to absorb arrival variance.
- **Statistical Multiplexing**: A dynamic channel-sharing strategy where link capacity is allocated on demand among active flows. Unlike fixed Time-Division Multiplexing (TDM) or Frequency-Division Multiplexing (FDM), statistical multiplexing maximizes link utilization for bursty data traffic, though it introduces queuing delay when instantaneous demand exceeds link bandwidth.

### 1.3 The End-to-End Argument

The End-to-End Argument is a foundational design principle of Internet architecture. It asserts that functions and controls should be implemented at the end-hosts (the application or transport endpoints) rather than inside the network core, unless implementing them in the core provides a critical performance optimization without compromising end-to-end correctness.

## Section 2: Framing & Error Handling

### 2.1 Framing Mechanics at the Link Layer

The physical layer delivers a continuous stream of raw bits. The link layer organizes these bits into discrete units called **frames** to enable synchronization, addressing, and error checking.

#### Framing Methods

- **Byte-Oriented Framing**:
    - _Sentinel-Based_: Uses special control bytes such as `STX` (Start of Text) and `ETX` (End of Text). To transmit data bytes that match sentinel codes within the payload, the sender uses **Byte Stuffing** (escaping), inserting a Data Link Escape (`DLE`) byte before any literal payload sentinel.
    - _Byte-Count-Based_: Includes an explicit length field in the frame header. If bit errors corrupt the length field, frame boundary synchronization is lost until a resynchronization signal occurs.
- **Bit-Oriented Framing**:
    - Treats the frame as a continuous bit stream regardless of byte boundaries (e.g., HDLC/SDLC).
    - Uses a unique 8-bit flag pattern `01111110` (`0x7E`) to denote frame start and end.
    - **Bit Stuffing Algorithm**: The transmitter monitors the payload bit stream. Whenever it detects five consecutive `1`s, it automatically inserts (stuffs) a `0` bit into the transmitted stream. Upon reception, whenever the receiver observes five consecutive `1`s followed by a `0`, it strips the stuffed `0` bit. If it sees six consecutive `1`s followed by a `0`, it recognizes a frame delimiter flag `01111110`.
- **Clock-Based Framing**:
    - Synchronous frameworks like SONET (Synchronous Optical Network) enforce a rigid timing structure, where frames occur in fixed 125 \mu\text{s} time intervals with dedicated framing bytes (A_1, A_2) for alignment.

### 2.2 Error Detection Code Mathematics

Physical communication channels suffer from thermal noise and burst interference, causing bit inversions. Error detection codes add redundant bits to a message frame to allow the receiver to detect bit errors.

#### Parity Schemes

- **Single Parity Bit**: Appends one bit to ensure the total number of `1` bits in the frame is even (or odd). Detects any odd number of bit errors, but fails completely when an even number of errors occurs.
- **Two-Dimensional Parity**: Arranges data bits into a two-dimensional grid and computes parity for each row and column. Detects all 1, 2, and 3-bit errors, and can correct single-bit errors by identifying the intersecting row and column.

#### Internet Checksum

Used extensively in IP, UDP, and TCP protocols.

- **Computation**: The sender divides the header/data into 16-bit integers, sums them using 1's complement addition (where carries beyond 16 bits are wrapped around and added to the least significant bit), and takes the 1's complement of the final sum.
- **Limitations**: Simple to implement in software, but vulnerable to bit transposition errors (reordered words yield identical sums) and insertion of zero-valued blocks.

#### Cyclic Redundancy Check (CRC)

CRC offers robust error detection resilience against burst errors using polynomial arithmetic modulo 2.

The table below summarizes the roles of the operational polynomials in CRC generation and verification:

|   |   |   |
|---|---|---|
|Polynomial / Parameter|Symbol|Definition / Operational Role|
|Message Bit Stream|D(x)|Binary polynomial representing raw payload data of length m bits|
|Generator Polynomial|g(x)|Pre-agreed divisor polynomial of degree k (requiring k+1 bits)|
|Shifted Message|x^k D(x)|Message polynomial shifted left by k bits (appends k zeros)|
|Frame Check Sequence|r(x)|Remainder polynomial of degree \le k-1 resulting from x^k D(x) / g(x)|
|Transmitted Frame|P(x)|Combined frame polynomial: P(x) = x^k D(x) \oplus r(x)|

#### Modulo-2 CRC Mathematical Workflow

1. Let the message be represented as a binary polynomial D(x) of degree m-1.
2. Select a fixed Generator polynomial g(x) of degree k (requiring k+1 bits).
3. Multiply D(x) by x^k (equivalent to appending k zero bits to the right of message D).
4. Divide x^k D(x) by g(x) using Modulo-2 division (XOR subtraction) to obtain a remainder polynomial r(x) of degree at most k-1: x^k D(x) = q(x) \cdot g(x) \oplus r(x)
5. The transmitted bit sequence is P(x) = x^k D(x) \oplus r(x), where r(x) is the Frame Check Sequence (FCS).
6. Upon receipt, the receiver divides P(x) by g(x). If no bit errors occurred, the remainder is identically 0.

#### Error Detection Bounds

A CRC with a generator polynomial of degree k guarantees detection of all single-bit errors, all double-bit errors (for appropriate generator choices), any odd number of bit errors, and any burst error of length \le k bits.

## Section 3: Reliable Transmission, Flow Control & Transport Layer

### 3.1 Automatic Repeat reQuest (ARQ) Protocols

When a receiver detects a corrupted frame via CRC, it discards the frame. Reliable transmission protocols retransmit missing or corrupted data using Automatic Repeat reQuest (ARQ) mechanisms combining Acknowledgements (ACKs), Sequence Numbers, and Retransmission Timeouts.

#### Stop-and-Wait ARQ

The simplest reliable protocol mechanism:

- **Operation**: The sender transmits a single frame, starts a timer, and pauses. The receiver returns an `ACK` frame upon valid receipt. The sender transmits the next frame only after receiving the `ACK`. If the timer expires before the `ACK` arrives, the sender retransmits the original frame.
- **Efficiency Analysis**: The maximum link utilization (U) of Stop-and-Wait on a channel with transmission delay T_{\text{trans}} and round-trip time \text{RTT} is: U = \frac{T_{\text{trans}}}{T_{\text{trans}} + \text{RTT}} On high-latency or high-bandwidth links (\text{RTT} \gg T_{\text{trans}}), Stop-and-Wait link utilization approaches zero, rendering it unusable for high-performance networks.

#### Sliding Window Protocols

Sliding window mechanisms pipeline multiple frames to fill the network pipe.

- **Send Window Size (****SWS****)**: The maximum number of unacknowledged frames the sender is permitted to transmit.
- **Receive Window Size (****RWS****)**: The maximum number of out-of-order frames the receiver is willing to buffer.
- **Sequence Number Bit-Width (****m****)**: Sequence numbers wrap around modulo 2^m. To avoid ambiguity between original frames and retransmissions, the window sizes must satisfy: SWS + RWS \le 2^m
    - **Go-Back-N Protocol**: RWS = 1. The receiver accepts frames strictly in sequence and discards out-of-order frames, sending cumulative ACKs. If a packet is lost, the sender's timeout forces retransmission of the lost frame and all subsequent unacknowledged frames. Requires SWS \le 2^m - 1.
    - **Selective Repeat Protocol**: RWS = SWS. The receiver buffers valid out-of-order frames and returns selective ACKs. The sender retransmits only the specifically unacknowledged frames. Requires SWS \le 2^{m-1}.

### 3.2 Transport Layer Mechanics & TCP Overview

While the Network Layer provides host-to-host packet delivery, the Transport Layer provides process-to-process communication using **Port Numbers** for application demultiplexing.

#### Flow Control vs. Congestion Control

- **Flow Control**: A point-to-point mechanism designed to protect the receiver's buffer space from being overflowed by a fast sender.
- **Congestion Control**: A system-wide mechanism designed to protect intermediate network routers and switches from buffer exhaustion caused by collective traffic overload.

#### TCP Segment Header Overview

The Transmission Control Protocol (TCP) is a connection-oriented, reliable, byte-stream transport protocol. Key header fields include:

- **Source Port (16 bits)** & **Destination Port (16 bits)**: Demultiplexing endpoints.
- **Sequence Number (32 bits)**: The byte-stream index of the first data byte in the segment.
- **Acknowledgement Number (32 bits)**: The next byte index expected by the sender (cumulative ACK).
- **Window Size (16 bits)**: Advertised Receiver Window (`RWIN`), explicitly communicating available receiver buffer space for flow control.
- **Control Flags**: `SYN` (synchronize sequence numbers), `ACK` (acknowledgement valid), `FIN` (finish connection), `RST` (reset connection), `PSH` (push data), `URG` (urgent data).

#### Connection Establishment: TCP Three-Way Handshake

1. **SYN**: Client sends a segment with `SYN=1`, random Initial Sequence Number x (`ISN_c`).
2. **SYN-ACK**: Server responds with `SYN=1`, `ACK=1`, `Ack_No = x + 1`, and server `ISN_s = y`.
3. **ACK**: Client responds with `ACK=1`, `Ack_No = y + 1`, `Seq_No = x + 1`. Connection enters `ESTABLISHED` state.

## Section 4: Congestion Control Principles & TCP Reno

### 4.1 Fundamentals of Network Congestion

Network congestion occurs when the aggregate arrival rate of packets at an intermediate switch or router exceeds its output link capacity. Excess packets accumulate in router buffers, increasing queuing delay. If overload persists, buffer space is exhausted (tail-drop), causing packet loss.

- **AIMD Principle**: Additive Increase, Multiplicative Decrease (AIMD) is the necessary control law for stable, fair, and efficient distributed congestion control.

### 4.2 TCP Reno Congestion Control Architecture

TCP Reno dynamically controls its sending rate via a state variable called the **Congestion Window (****cwnd****)**. The effective transmission window is bounded by both flow and congestion control:

\text{Effective Window} = \min(cwnd, \text{RWIN})

The operational phases and state machine transitions of TCP Reno are summarized in the table below:

|   |   |   |   |
|---|---|---|---|
|Phase / Event|Trigger Condition|State Variable Action|Growth / Rate Trajectory|
|**Slow Start**|Connection initiation or RTO timeout|cwnd = cwnd + 1 \text{ MSS} per ACK|Exponential growth (cwnd doubles every RTT)|
|**Congestion Avoidance**|cwnd \ge ssthresh|cwnd = cwnd + \frac{\text{MSS}}{cwnd} per ACK|Linear growth (+1 \text{ MSS} added per RTT)|
|**Timeout (RTO)**|Retransmission timer expires|ssthresh = \frac{cwnd}{2}, cwnd = 1 \text{ MSS}|Resets to Slow Start|
|**Fast Retransmit**|Arrival of 3 duplicate ACKs|Retransmits missing segment immediately|Bypasses RTO timer delay|
|**Fast Recovery**|Post 3 duplicate ACKs|ssthresh = \frac{cwnd}{2}, cwnd = ssthresh + 3|Retains linear probing without Slow Start drop|

#### Detailed State Machine Rules

1. **Slow Start Phase**:
    - _Goal_: Rapidly probe for available bandwidth upon startup or recovery.
    - _Initialization_: Set cwnd = 1 \text{ MSS} (Maximum Segment Size).
    - _Growth Law_: For every received ACK, increment cwnd by 1 \text{ MSS}. This results in **exponential growth**, doubling cwnd every Round-Trip Time (\text{RTT}).
    - _Exit Condition_: When cwnd \ge ssthresh (slow start threshold), transition to Congestion Avoidance.
2. **Congestion Avoidance Phase**:
    - _Goal_: Cautiously probe for additional bandwidth while operating near channel capacity.
    - _Growth Law_: For each received ACK, cwnd = cwnd + \frac{\text{MSS}}{cwnd}. This results in **linear growth**, adding 1 \text{ MSS} to cwnd per \text{RTT}.
3. **Retransmission Timeout (RTO) Event**:
    - Indicates severe network congestion (no ACKs returning).
    - _Action_: Set ssthresh = \max\left(\frac{cwnd}{2}, 2 \text{ MSS}\right), reset cwnd = 1 \text{ MSS}, and re-enter **Slow Start**.
4. **Fast Retransmit & Fast Recovery**:
    - _3 Duplicate ACKs Signal_: Indicates that a single packet was lost, but subsequent packets arrived safely at the receiver (triggering duplicate ACKs for the missing sequence number).
    - _Fast Retransmit_: Immediately retransmit the missing segment without waiting for the RTO timer to expire.
    - _Fast Recovery_: Avoids dropping back to Slow Start. Sets ssthresh = \frac{cwnd}{2}, sets cwnd = ssthresh + 3 \text{ MSS}, and stays in linear congestion avoidance once the lost packet is acknowledged.

## Section 5: Packet Switching, Bridging & Local Area Networks

### 5.1 Switching Paradigms

1. **Circuit Switching**: Sets up a dedicated, physical end-to-end communication path prior to data transmission (e.g., traditional PSTN telephone networks). Guarantees fixed bandwidth and zero queuing delay, but wastes capacity during silent periods.
2. **Packet Switching**: Divides data streams into discrete packets that are routed through shared network links using **Store-and-Forward** processing. Intermediate switches buffer incoming packets, inspect destination headers, and queue packets for transmission on the appropriate output link. Achieves high efficiency through statistical multiplexing, but introduces queuing delays and potential packet drop.

### 5.2 Link-Layer Bridging & Self-Learning Switches

To connect multiple physical link segments into a single Local Area Network (LAN) without extending collision domains, networks use Layer 2 Switches (Bridges).

#### Self-Learning Algorithm

A Layer 2 switch operates transparently without manual configuration. It dynamically builds a **Filtering Database / MAC Table** mapping MAC addresses to switch ports (`[MAC Address, Port, Timestamp]`).

#### Switch Processing Rules

For each frame received on Port P with Source MAC S and Destination MAC D:

1. **Learn**: Entry [S, P] is created or updated in the MAC Table with the current timestamp.
2. **Lookup Destination** **D** **in MAC Table**:
    - If D is found on Port P_{\text{dest}}:
        - If P_{\text{dest}} == P \rightarrow Drop frame (filter; destination is on the same segment).
        - Else \rightarrow Forward frame exclusively to Port P_{\text{dest}} (unicast forwarding).
    - If D is NOT found in table, or is Broadcast (`FF:FF:FF:FF:FF:FF`):
        - \rightarrow Flood frame to ALL ports except incoming Port P.

#### Spanning Tree Protocol (STP / IEEE 802.1D)

Redundant physical links between switches create physical loops. In a looped network, flooded broadcast or unknown-destination frames circulate indefinitely, causing catastrophic **Broadcast Storms** and MAC table instability.

To prevent loops while maintaining physical redundancy, switches execute the distributed **Spanning Tree Protocol**:

1. **Root Bridge Election**: Switches exchange Bridge Protocol Data Units (BPDUs) to elect the switch with the numerically lowest **Bridge ID** (Priority + MAC Address) as the Root Bridge.
2. **Root Port Selection**: Each non-root switch identifies its Root Port—the port providing the lowest path-cost to the Root Bridge.
3. **Designated Port Selection**: On each network link, the switch presenting the lowest path cost to the root is assigned the Designated Port.
4. **Blocking State**: All remaining ports that are neither Root Ports nor Designated Ports are placed into a **Blocking State**, disabling frame forwarding to break all operational loops.

## Section 6: Network Layer Architecture, IPv4 Addressing & CIDR

### 6.1 The Internet Protocol (IP) Role

The Network Layer provides a unified, connectionless, best-effort packet delivery service across heterogeneous physical networks. This "Hourglass Architecture" places IP at the focal waist connecting diverse physical link technologies below with diverse transport protocols above.

- **Delivery Modes**:
    - _Unicast_: One-to-one host communication.
    - _Broadcast_: One-to-all hosts on a local subnet.
    - _Multicast_: One-to-many subscriber group delivery.

### 6.2 IPv4 Addressing Structure & CIDR

An IPv4 address is a 32-bit binary identifier written in dotted-decimal format (e.g., `192.168.1.1`).

#### Historical Classful Addressing

Originally divided into rigid allocation classes based on leading prefix bits:

- **Class A**: Leading bit `0`, 8-bit Network ID, 24-bit Host ID (`0.0.0.0/8` to `127.255.255.255/8`).
- **Class B**: Leading bits `10`, 16-bit Network ID, 16-bit Host ID (`128.0.0.0/16` to `191.255.255.255/16`).
- **Class C**: Leading bits `110`, 24-bit Network ID, 8-bit Host ID (`192.0.0.0/24` to `223.255.255.255/24`).

Classful addressing led to rapid address space exhaustion and routing table bloat due to its coarse allocation blocks.

#### Classless Inter-Domain Routing (CIDR)

CIDR eliminated class boundaries, allowing variable-length network prefixes denoted by prefix length `/p` (e.g., `128.230.12.0/22`).

#### Subnet Mask Calculations

A subnet mask is a 32-bit value containing p continuous leading `1` bits followed by 32-p zero bits.

\text{Usable Hosts} = 2^{32-p} - 2

_(Subtracting 2 for the Network Address where all host bits are_ `_0_`_, and the Subnet Broadcast Address where all host bits are_ `_1_`_)._

An example CIDR breakdown for prefix `192.168.16.0/20` is shown below:

```
Prefix Length p = 20 bits. Host Bits = 32 - 20 = 12 bits.
Subnet Mask: 255.255.240.0 (11111111.11111111.11110000.00000000)
Total IP Addresses = 2^12 = 4096.
Usable Hosts = 4096 - 2 = 4094.
Network Address: 192.168.16.0
Broadcast Address: 192.168.31.255
```

#### Route Aggregation (Supernetting)

CIDR allows upstream ISPs to aggregate contiguous customer prefixes into a single summary prefix announcement to reduce global BGP routing table bloat. For example, four contiguous `/24` networks (`200.23.16.0/24`, `200.23.17.0/24`, `200.23.18.0/24`, `200.23.19.0/24`) are aggregated into a single `/22` route: `200.23.16.0/22`.

## Section 7: IP Forwarding, ARP & Middleboxes (NAT/IPv6)

### 7.1 Router IP Forwarding & Longest Prefix Match

When an IP router receives a packet on an input interface, it extracts the Destination IP Address and queries its **Forwarding Information Base (FIB)** to select the appropriate output interface and next-hop router.

#### Longest Prefix Match (LPM) Rule

A forwarding table contains prefix/length tuples that may overlap. To resolve overlapping matches, routers enforce the **Longest Prefix Match Rule**: the router evaluates the destination address against all matching entries and selects the route entry with the **longest prefix length** **p** (most specific network mask).

Consider a FIB containing the entries in the table below:

|   |   |   |   |
|---|---|---|---|
|Destination Prefix / Mask|Next Hop / Output Interface|Match Evaluation for Destination `10.1.130.45`|Match Status|
|`10.1.0.0/16`|Interface 1|First 16 bits match `10.1`|Valid Match (Length 16)|
|`10.1.128.0/18`|Interface 2|First 18 bits match `10.1.128` (130 \text{ AND } 192 = 128)|Valid Match (Length 18)|
|`0.0.0.0/0` (Default)|Interface 3|Default fallback|Valid Match (Length 0)|

_Selection_: Entry `10.1.128.0/18` is selected because prefix length 18 > 16. The packet is forwarded to Interface 2.

### 7.2 Address Resolution Protocol (ARP)

IP forwarding determines the next-hop IP address, but link-layer Ethernet switches require the destination's 48-bit physical MAC address.

#### ARP Operational Sequence

1. **ARP Query**: When host A needs to deliver an IP packet to local IP B, it checks its local ARP Cache. If IP B is missing, Host A broadcasts an `ARP Request` packet (`"Who has IP B? Tell MAC A"`) to Ethernet broadcast address `FF:FF:FF:FF:FF:FF`.
2. **ARP Reply**: Target Host B receives the broadcast query, updates its own ARP table with Host A's mapping, and returns a unicast `ARP Reply` packet (`"IP B is at MAC B"`) directly to Host A.
3. **Caching**: Host A records IP B \rightarrow MAC B in its ARP cache with a Time-To-Live (TTL) expiration timer.

### 7.3 Network Address Translation (NAT) & IPv6

To mitigate IPv4 address depletion, middleboxes implement Network Address Translation.

#### Private IP Address Spaces (RFC 1918)

- `10.0.0.0/8` (`10.0.0.0` to `10.255.255.255`)
- `172.16.0.0/12` (`172.16.0.0` to `172.31.255.255`)
- `192.168.0.0/16` (`192.168.0.0` to `192.168.255.255`)

Private addresses are non-routable over the public global Internet.

#### Port Address Translation (PAT / NAPT) Mechanics

A NAT router maps multiple private local IP addresses behind a single public WAN IP address using transport-layer port numbers.

1. **Outgoing Packet (Internal Host** **\rightarrow** **Web Server)**:
    - Private IP Header: Source = `10.0.0.4:3345`, Destination = `132.239.8.45:80`.
    - NAT Action: Rewrites Source IP to public WAN IP `138.76.29.7` and assigns public Port `5001`.
    - NAT Translation Table Entry Recorded: `[138.76.29.7:5001 <---> 10.0.0.4:3345]`.
    - Transmitted Packet: Source = `138.76.29.7:5001`, Destination = `132.239.8.45:80`.
2. **Incoming Reply (Web Server** **\rightarrow** **NAT Router)**:
    - Packet Arrives: Source = `132.239.8.45:80`, Destination = `138.76.29.7:5001`.
    - NAT Action: Queries translation table for `138.76.29.7:5001` \rightarrow Maps to `10.0.0.4:3345`.
    - Rewrites Destination IP to `10.0.0.4` and Destination Port to `3345`.
    - Forwarded to LAN Host `10.0.0.4`.

#### NAT Architectural Challenges

- Breaks end-to-end IP address reachability (incoming unsolicited connections from the Internet are dropped unless static port forwarding or NAT Traversal protocols like STUN/ICE are implemented).
- Violates layering by inspecting and modifying transport-layer headers (ports) within the network layer.

#### IPv6 Architecture

IPv6 provides a definitive solution to address exhaustion:

- **128-Bit Addresses**: Yielding 2^{128} \approx 3.4 \times 10^{38} unique addresses, written in 8 hexadecimal colon-separated fields (e.g., `2001:0db8:85a3:0000:0000:8a2e:0370:7334`).
- **Simplified Fixed 40-Byte Header**: Eliminates header checksums to accelerate router forwarding, replaces IP options with chaining extension headers, and removes in-network router fragmentation (requiring path MTU discovery by hosts).

## Section 8: Application Layer Naming & Media Streaming

### 8.1 Domain Name System (DNS)

DNS is a distributed, hierarchical database that translates human-readable domain names (e.g., `www.ucsd.edu`) into IP addresses.

#### Hierarchical Database Structure

1. **Root Name Servers (**`**.**`**)**: 13 logical root server authorities globally managing top-level domain pointers.
2. **Top-Level Domain (TLD) Servers**: Manage domains under specific extensions (`.com`, `.edu`, `.org`, `.gov`).
3. **Authoritative Name Servers**: Maintained by organizations to provide definitive hostname-to-IP mappings for their specific domain space (e.g., `ucsd.edu`).

#### Resolution Query Modes

- **Iterative Resolution**: The queried DNS server returns the best available referral pointer to the next lower server in the hierarchy, requiring the querying client/local server to send sequential follow-up queries.
- **Recursive Resolution**: The queried server assumes full responsibility for resolving the name, querying downstream servers recursively on behalf of the requester until returning the final answer.

#### Common DNS Resource Records (RR)

A DNS entry is stored as a tuple `(Name, Value, Type, TTL)`:

- **Type A**: Name = Hostname, Value = IPv4 Address.
- **Type AAAA**: Name = Hostname, Value = IPv6 Address.
- **Type CNAME**: Name = Alias, Value = Canonical Real Hostname.
- **Type MX**: Name = Domain, Value = Mail Server Hostname.
- **Type NS**: Name = Domain, Value = Authoritative Name Server Hostname.

### 8.2 Media Streaming & DASH Architecture

Video streaming accounts for the vast majority of Internet bandwidth consumption.

#### Dynamic Adaptive Streaming over HTTP (DASH)

To deliver smooth video playback over unpredictable Internet paths subject to variable throughput and jitter, modern applications use DASH.

1. **Video Encoding**: Video content is encoded at multiple bitrate and resolution levels (e.g., 500 kbps, 2 Mbps, 8 Mbps).
2. **Segmentation**: Each video encoding is chunked into short, independent 2–10 second media segments (typically MP4 fragments).
3. **Media Presentation Description (MPD)**: An XML manifest file provided to the client at stream start. The MPD lists available video representations, resolutions, bitrates, codecs, and segment URL patterns.
4. **Client Adaptation Logic**: The client HTTP player continuously measures current download rates and buffer occupancy. If throughput drops or the buffer depletes, the client requests lower-bitrate segments. When network conditions improve, it steps up to higher-quality segments, avoiding playback stalls.

## Section 9: Short-Answer Review Quiz

Test your understanding with these 10 review questions covering core calculations and concepts:

1. **Latency Calculation**: Calculate the total transmission delay and propagation delay for a 1,500-byte packet transmitted over a 100 Mbps fiber-optic link with a length of 1,000 km (C = 2 \times 10^8 \text{ m/s}).
2. **Bit Stuffing**: Perform HDLC bit-stuffing on the following raw binary payload: `011111101111111101`.
3. **ARQ Efficiency**: On a link with a Round-Trip Time of 80 ms and a transmission delay of 0.2 ms, what is the maximum achievable link utilization using Stop-and-Wait ARQ?
4. **Sliding Window Constraints**: If a protocol uses 4-bit sequence numbers (m=4), what is the maximum allowable Send Window Size (SWS) for Go-Back-N ARQ? What is it for Selective Repeat ARQ?
5. **TCP Sequence Numbers**: A TCP sender transmits a segment with Sequence Number 5,000 containing 1,000 bytes of payload data. What Acknowledgement Number will the receiver return upon successful receipt?
6. **TCP Congestion Control Dynamics**: A TCP Reno connection currently in Congestion Avoidance has cwnd = 16 \text{ MSS} and ssthresh = 16 \text{ MSS}. If 3 duplicate ACKs arrive, what are the new values for ssthresh and cwnd?
7. **Subnetting Analysis**: Given the IP address `172.16.35.70/20`, determine the Subnet Mask, Network Address, Broadcast Address, and total number of usable host addresses.
8. **Longest Prefix Match**: A router has FIB entries: `10.1.0.0/16` (Port 1), `10.1.128.0/18` (Port 2), and `0.0.0.0/0` (Port 3). To which port will a packet with destination `10.1.130.5` be forwarded? Show the binary mask comparison.
9. **NAT Translation**: A local host `10.0.0.15:4500` sends a TCP packet to a web server `140.247.50.1:80` through a NAT router with public IP `203.0.113.5`. Describe the header source and destination modifications on both outgoing and incoming packets.
10. **DNS Query Types**: Explain the operational difference between Recursive and Iterative DNS queries from the perspective of the Local Name Server.

## Section 10: Quiz Answer Key & Worked Solutions

1. **Latency Solution**:
    - L = 1500 \text{ bytes} = 12,000 \text{ bits}.
    - R = 100 \text{ Mbps} = 10^8 \text{ bits/sec}.
    - T_{\text{trans}} = \frac{12,000}{10^8} = 0.00012 \text{ seconds} = 0.12 \text{ ms}.
    - D = 1,000 \text{ km} = 10^6 \text{ meters}, C = 2 \times 10^8 \text{ m/s}.
    - T_{\text{prop}} = \frac{10^6}{2 \times 10^8} = 0.005 \text{ seconds} = 5.0 \text{ ms}.
    - _Total Base Delay_ (excluding queuing/processing) = 5.0 \text{ ms} + 0.12 \text{ ms} = 5.12 \text{ ms}.
2. **Bit Stuffing Solution**:
    - Raw Input: `0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1`
    - Scan for five consecutive `1`s:
        - `0 1 1 1 1 1` \rightarrow insert `0` \rightarrow `0 1 1 1 1 1` `**0**` `1 0 ...`
        - Next block: `1 1 1 1 1 1 1 1` \rightarrow after first 5 ones, insert `0`: `1 1 1 1 1` `**0**` `1 1 1 0 1`
    - Stuffed Output: `01111101011111011101`.
3. **ARQ Efficiency Solution**:
    - U = \frac{T_{\text{trans}}}{T_{\text{trans}} + \text{RTT}} = \frac{0.2 \text{ ms}}{0.2 \text{ ms} + 80 \text{ ms}} = \frac{0.2}{80.2} \approx 0.00249 (or 0.25\% utilization).
4. **Sliding Window Constraints Solution**:
    - Sequence bits m = 4, total sequence space 2^4 = 16.
    - For Go-Back-N (RWS = 1): SWS \le 2^m - 1 = 15.
    - For Selective Repeat (RWS = SWS): SWS \le 2^{m-1} = 8.
5. **TCP Sequence Numbers Solution**:
    - The segment occupies byte sequence numbers 5,000 through 5,999.
    - The receiver returns an ACK expecting the next byte: `Ack_No = 6000`.
6. **TCP Congestion Control Dynamics Solution**:
    - Arrival of 3 duplicate ACKs triggers Fast Retransmit / Fast Recovery.
    - New ssthresh = \frac{cwnd}{2} = \frac{16}{2} = 8 \text{ MSS}.
    - New cwnd = ssthresh + 3 \text{ MSS} = 8 + 3 = 11 \text{ MSS}.
7. **Subnetting Analysis Solution**:
    - Prefix `/20` \rightarrow Mask = `255.255.240.0` (`11111111.11111111.11110000.00000000`).
    - IP `172.16.35.70`: 3rd octet = 35 = 00100011_2.
    - Mask 3rd octet = 240 = 11110000_2.
    - Network 3rd octet = 35 \text{ AND } 240 = 32 (00100000_2).
    - **Network Address**: `172.16.32.0`.
    - Host bits = 12 \rightarrow Host range spans 3rd octet 32 to 47.
    - **Broadcast Address**: `172.16.47.255`.
    - **Usable Host Range**: `172.16.32.1` to `172.16.47.254`.
    - **Total Usable Hosts**: 2^{12} - 2 = 4094.
8. **Longest Prefix Match Solution**:
    - Destination `10.1.130.5`:
        - Binary 3rd octet = 130 = 10000010_2.
    - Evaluate `10.1.0.0/16`: First 16 bits match `10.1` \rightarrow Valid Match.
    - Evaluate `10.1.128.0/18`: Subnet mask `/18` covers 3rd octet top 2 bits (`11000000_2` = 192).
        - 130 \text{ AND } 192 = 128 (10000000_2).
        - Matches prefix `10.1.128.0/18` \rightarrow Valid Match.
    - Longest Match Rule selects prefix `/18` (length 18 > 16).
    - Forwarded to **Port 2**.
9. **NAT Translation Solution**:
    - _Outgoing Packet (Host_ _\rightarrow_ _NAT)_: Src = `10.0.0.15:4500`, Dst = `140.247.50.1:80`.
    - _NAT Translation_: Replaces Src IP/Port with public WAN allocation, e.g., `203.0.113.5:61001`.
    - _Outgoing Packet (NAT_ _\rightarrow_ _Server)_: Src = `203.0.113.5:61001`, Dst = `140.247.50.1:80`.
    - _Incoming Reply (Server_ _\rightarrow_ _NAT)_: Src = `140.247.50.1:80`, Dst = `203.0.113.5:61001`.
    - _NAT Translation_: Rewrites Dst IP/Port back to internal mapping `10.0.0.15:4500`.
    - _Incoming Packet (NAT_ _\rightarrow_ _Host)_: Src = `140.247.50.1:80`, Dst = `10.0.0.15:4500`.
10. **DNS Query Types Solution**:
    - In a **Recursive Query**, the client requests that the server handle the full resolution path. The queried DNS server must locate and return the final IP mapping (or an error), querying other authoritative servers on the client's behalf.
    - In an **Iterative Query**, the queried server returns the best referral answer it holds in its cache or hierarchy (e.g., pointing to the TLD or authoritative server), requiring the client itself to issue the follow-up query to the next server.

## Section 11: Comprehensive Synthesis & Discussion Questions

These discussion questions require synthesizing concepts across multiple functional layers:

1. **Cross-Layer Interaction & The End-to-End Argument**: Explain why error detection and retransmission mechanisms are often implemented at both the Link Layer (e.g., CRC and ARQ in 802.11 Wi-Fi) and the Transport Layer (e.g., TCP Checksum and ARQ). Does this duplication violate the End-to-End Argument? Justify your answer.
2. **Bandwidth-Delay Product & TCP Window Scaling**: Consider a 10 Gbps transcontinental fiber link with an RTT of 100 ms. Calculate the BDP. Why does standard TCP header architecture (with its 16-bit Window Size field) fail to utilize this link effectively without the TCP Window Scaling option?
3. **Congestion Control vs. Media Streaming**: Modern HTTP video streaming (DASH) runs over TCP. Describe the interaction between TCP Reno's AIMD congestion control dynamics (with its saw-tooth throughput variations) and the video client's playback buffer. How does client-side bitrate adaptation prevent video freezing during TCP multiplicative decrease events?
4. **Network-Layer vs. Link-Layer Forwarding**: Compare and contrast the forwarding lookup mechanisms of a Layer 2 Self-Learning Switch and a Layer 3 IP Router. Highlight differences in address structure, lookup algorithms (Exact Matching vs. Longest Prefix Matching), and handling of loops (STP vs. IP TTL).
5. **Middlebox Architectural Conflicts**: Discuss how Network Address Translation (NAT) and Firewalls interact with peer-to-peer (P2P) networking protocols and IPsec security headers. Why do middleboxes complicate the traditional IP architecture model?

## Section 12: Glossary of Essential Computer Networking Terms

- **Additive Increase Multiplicative Decrease (AIMD)**: A linear control law used in TCP congestion control that increments window size linearly during congestion avoidance and cuts window size multiplicatively upon loss.
- **Address Resolution Protocol (ARP)**: A link-layer protocol that dynamically maps 32-bit IPv4 addresses to 48-bit physical Ethernet MAC addresses.
- **Bandwidth-Delay Product (BDP)**: The volume of unacknowledged data required to keep a network link fully saturated, calculated as R \times \text{RTT}.
- **Bridge Protocol Data Unit (BPDU)**: Management frame exchanged between Layer 2 switches to execute the Spanning Tree Protocol (STP).
- **Classless Inter-Domain Routing (CIDR)**: An IP addressing scheme that replaces classful boundaries with variable-length network prefixes (`/p`).
- **Congestion Window (****cwnd****)**: A TCP sender-side state variable regulating transmission volume to prevent network core buffer overload.
- **Cyclic Redundancy Check (CRC)**: A polynomial error detection code operating modulo 2 that detects single, double, odd, and burst errors.
- **Dynamic Adaptive Streaming over HTTP (DASH)**: An application protocol that streams chunked video over HTTP, dynamically adjusting video bitrate based on client throughput measurements.
- **Domain Name System (DNS)**: A distributed hierarchical database system that translates domain hostnames into IP addresses.
- **End-to-End Argument**: An architectural principle stating that network functions should be placed at application endpoints rather than inside intermediate network core nodes.
- **Fast Retransmit**: A TCP Reno optimization that retransmits a missing segment immediately upon receiving 3 duplicate ACKs without waiting for a timeout.
- **Fast Recovery**: A TCP Reno algorithm that sets cwnd = ssthresh after Fast Retransmit, maintaining linear transmission without dropping to Slow Start.
- **Forwarding Information Base (FIB)**: A routing/forwarding table in a router mapping IP destination prefixes to output interfaces and next-hop addresses.
- **Jitter**: Packet delay variance across consecutive packet arrivals.
- **Longest Prefix Match (LPM)**: The forwarding lookup rule that selects the FIB entry with the most specific (longest) network mask.
- **Maximum Segment Size (MSS)**: The maximum TCP payload size in bytes that a host can send in a single unfragmented IP packet.
- **Media Presentation Description (MPD)**: An XML manifest file used in DASH streaming describing available segment URLs, resolutions, and bitrates.
- **Network Address Translation (NAT)**: A middlebox mechanism mapping private local IP addresses and ports to public global IP addresses and ports.
- **Propagation Delay**: Time required for a physical signal to travel across a physical medium (D / C).
- **Round-Trip Time (RTT)**: Total time required for a packet to travel from sender to receiver and return an acknowledgment.
- **Self-Learning Switch**: A Layer 2 switch that inspects incoming frame source MAC addresses to dynamically populate its forwarding table.
- **Slow Start**: The initial TCP congestion control phase where cwnd increases exponentially, doubling every RTT.
- **Spanning Tree Protocol (STP)**: A distributed network protocol (IEEE 802.1D) that disables redundant switch ports to eliminate operational loops in Layer 2 networks.
- **Statistical Multiplexing**: Dynamic bandwidth sharing among multiple active flows based on instantaneous demand.
- **Transmission Delay**: Time required to push all bits of a packet onto the physical medium (L / R).