---
title: "Digital Communication" 
description: "Index for Digital Communication, covering network socket I/O, HTTP communication protocols, state management, and HTTP path parsing." 
aliases: 
- Digital Communication Hub 
- Network IO & HTTP 
tags: 
- index 
- system-programming 
- networking 
- http 
---

> [!ABSTRACT]
> 
> This index explores how computers identify each other, the language they use to request information (URLs), and the physical path data takes across the internet.
### 1. The URL (Uniform Resource Locator)
A URL is more than just a web address; it is a structured command sent to a remote server.

Structure Breakdown:

`https://www.google.com:443/search?q=ucsd+spring`

|**Component**|**Example**|**Description**|
|---|---|---|
|**Protocol**|`http` / `https`|The "language" or syntax used to send data. The `s` in `https` denotes a secure connection using SSL/TLS.|
|**Domain**|`www.google.com`|The human-readable name of the server (mapped to an IP address via DNS).|
|**Port**|`:8000` / `:443`|Specifies which "door" to enter on the server. Protocols have defaults: HTTP is `80`, HTTPS is `443`.|
|**Path**|`/search`|The specific resource or file being requested on the server.|
|**Query**|`?q=ucsd+spring`|Key-value pairs providing arguments to the server logic (e.g., search terms).|

---
### 2. Network Protocols: HTTP vs. HTTPS
The protocol determines how data is "wrapped" for travel.
- **HTTP (Hypertext Transfer Protocol):** Data is sent in "plain text." Anyone on the same network (like a public Wi-Fi) can potentially intercept and read the data.
- **HTTPS (Secure):** Data is encrypted. Even if intercepted, the content looks like gibberish without the private security keys managed by domain authorities.

---
### 3. Why Networking? (The Shared Infrastructure)
Computers do not have dedicated wires to every other computer on earth. Instead, they use a shared infrastructure of routers and switches.
#### Tracking the Path: `traceroute`
To see the "hops" your data takes to reach a destination, use the `traceroute` command. Each hop represents a router or gateway that directs your data closer to the target.

```Bash
traceroute ieng6.ucsd.edu
# Shows the sequence of IP addresses and the time (latency) 
# it takes to jump from your computer to the server.
```

---
### 4. Digital Communication Concepts Table

| **Concept**    | **Summary**                                                                |
| -------------- | -------------------------------------------------------------------------- |
| **IP Address** | The numerical "phone number" of a computer (e.g., `128.54.x.x`).           |
| **DNS**        | The "Phonebook" of the internet that turns `google.com` into an IP.        |
| **Latency**    | The delay (usually in milliseconds) for a packet to travel between points. |
| **HTML**       | The standard markup language used to describe the structure of a webpage.  |

---
### 5. Module Implementation & Reference Files
The following files represent the transition from these networking theories into actual C implementation:
- **[[Network IO & HTTP Communication]]**: Focuses on the system-level mechanics—sockets, function pointers, and the `write()` system call.
- **[[HTTP Server - State and Path Parsing]]**: Focuses on the application logic—handling persistent global state and parsing URL queries.