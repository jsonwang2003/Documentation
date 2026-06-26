---
title: Problem Sets 5
---
> [!ABSTRACT]
> 
> This problem set marks the culmination of the course, integrating Socket Programming, String Parsing, and Systems Architecture. It focuses on building a functional HTTP server from scratch, handling network requests, and managing dynamic web content.

---
## Problem Categories
### Socket Programming & Server Basics
- **[[Hello World Server]]**: The "minimum viable product" for networking—setting up a listener and sending a simple string to a client.
- **[[Echo Server]]**: A server that reads data from a socket and writes it back, demonstrating basic two-way communication.
- **[[Status Code Server]]**: Implementing the logic to return specific HTTP status codes (e.g., 200 OK, 404 Not Found) based on request validity.
- **[[Timestamp Functions]]**: Generating correctly formatted date/time strings required for HTTP response headers.
### HTTP Request Parsing
- **[[Recognize HTTP Path]]**: Identifying the requested resource from the initial line of an HTTP request.
- **[[URL Decode String]]**: Converting percent-encoded URLs (e.g., `%20` to space) back into standard characters.
- **[[Count Query Parameters]]**: Parsing the `?key=value` portion of a URL to determine how much data is being passed.
- **[[Extract ID from React Request]]**: Handling modern front-end routing by pulling specific identifiers from incoming paths.
- **[[Extract Message from Request Path]]**: A utility for parsing custom data embedded within the URL structure.
### POST Requests & Data Extraction
- **[[Extract Username from Post Request]]**: Digging into the body of an HTTP POST request to retrieve form data.
- **[[UTF-8 Decoder]]**: Ensuring that text received over the network is correctly interpreted using multi-byte character logic.
- **[[SHA256 Hash Program]]** / **[[SHA256 Hash Program (Hex)]]**: Using hashing to verify data or process passwords received via the server.
### Application Logic (The Chat System)
- **[[Format Single Chat Line]]**: Transforming raw message data into a structured format for display.
- **[[Format Reaction Line]]**: Handling social features like emoji reactions by generating specific HTML or JSON responses.
- **[[Calculator Server]]**: A dynamic server that performs arithmetic based on parameters provided in the HTTP request.

---
## Technical Summary

|**Step**|**System Function**|**Purpose**|
|---|---|---|
|**Setup**|`socket()`, `bind()`|Create a communication endpoint and assign it a port.|
|**Listen**|`listen()`, `accept()`|Wait for an incoming connection and create a dedicated client socket.|
|**Process**|`read()`, `parse()`|Receive the raw HTTP request and extract the method/path.|
|**Respond**|`write()`, `close()`|Send the headers and body back to the client, then terminate the connection.|

---
## Related Toolkits
- **[[Syntax/index|C Syntax]]**: Critical for managing buffers and string pointers during request parsing.
- **[[Practice Examples/Lecture Examples/Digital Communication/index|Lecture: Digital Communication]]**: The theoretical foundation for TCP, IP, and the HTTP protocol.
- **[[Practice Examples/Problem Sets 1/index|Problem Sets 1: Bits and Encodings]]**: Re-applied here for the UTF-8 decoding of network-transmitted text.