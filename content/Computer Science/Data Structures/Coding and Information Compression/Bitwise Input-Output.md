## Introduction

In lossless data compression (like [[Huffman Code]]), we often generate encoded data in sequences of bits that do not align perfectly with byte boundaries. However, physical storage devices and operating systems are designed to handle data in **bytes** (8-bit chunks). **Bitwise I/O** is the process of bridging this gap, allowing us to read and write data bit-by-bit while adhering to the byte-oriented nature of hardware.

---
## The Buffering Strategy

Accessing a disk is an extremely slow operation compared to memory. To maintain performance, programming languages use **buffers**—temporary storage areas in fast memory (RAM).
1. **Bytewise Buffer:** A standard memory block (typically 4 KB). Data is collected here and written to disk in a single "flush" once the buffer is full.
2. **Bitwise Buffer:** A smaller, 1-byte (`unsigned char`) layer we build on top of the bytewise buffer. It collects 8 individual bits before sending the completed byte to the larger bytewise buffer.

### Writing Workflow:

#### High Level
![](https://ucarecdn.com/8c7bcd71-346c-496f-9090-5892c5e89534/)

#### Mid Level
- Write bits to the **bitwise buffer** one at a time.
- Once 8 bits are collected, **flush** the bitwise buffer (write the byte) to the **bytewise buffer**.
- Once the bytewise buffer is full (e.g., 4,096 bytes), **flush** it to the **disk**.

![](https://ucarecdn.com/70497dcb-2bf0-4445-86dc-968ee12c9671/ "Image: https://ucarecdn.com/70497dcb-2bf0-4445-86dc-968ee12c9671/")
### Reading Workflow:
#### High Level
![](https://ucarecdn.com/4c6f8f56-1309-4995-8f5a-8f706c65112d/ "Image: https://ucarecdn.com/4c6f8f56-1309-4995-8f5a-8f706c65112d/")

#### Mid Level
- **Fill** the bytewise buffer by reading a large block from the disk.
- **Fill** the bitwise buffer by pulling 1 byte from the bytewise buffer.
- Read individual bits from the bitwise buffer.

![](https://ucarecdn.com/682d3290-b026-4e47-820c-878e3f28b65d/ "Image: https://ucarecdn.com/682d3290-b026-4e47-820c-878e3f28b65d/")

---
## The Problem of Padding

Because the smallest unit writable to disk is 1 byte, we face a challenge if our message ends and the bitwise buffer isn't full.

**Example:** You want to write 12 bits: `11111111 1111`.
1. The first 8 bits (`11111111`) are written as a full byte.
2. The remaining 4 bits (`1111`) are stuck in the buffer.
3. **Solution:** We "pad" the byte with 0s to make it 8 bits: `11110000`.

**The Ambiguity:** When reading back `11110000`, how does the computer know if the last four zeros are actual data or just padding?

**The Solution (Headers):** We include a **header** at the start of the file. This metadata tells the program how many bits to expect. For our 12-bit example, a header might store the integer `12`, telling the reader to stop after exactly 12 bits and ignore any remaining padding in the final byte.

---
## Implementation in C++

To handle this automatically, we design two classes: `BitOutputStream` and `BitInputStream`.

### BitOutputStream

This class wraps a standard `ostream` to provide bit-level writing.

```cpp
class BitOutputStream {
    private:
        unsigned char buf; // 8-bit buffer
        int nbits;         // current bit count (0-8)
        ostream & out;     // the underlying byte-stream

    public:
        BitOutputStream(ostream & os) : out(os), buf(0), nbits(0) {}

        void flush() {
            out.put(buf);  // Send the byte to the byte-buffer
            buf = 0;       
            nbits = 0;     
        }

        void writeBit(unsigned int bit) {
            if(nbits == 8) flush();
            
            // Logic to set the specific bit in the buffer
            // Example: buf |= (bit << (7 - nbits));
            nbits++;
        }
};
```

### BitInputStream

This class wraps a standard `istream` to provide bit-level reading.

```cpp
class BitInputStream {
    private:
        unsigned char buf;  // 8-bit buffer
        int nbits;          // bits already read (0-8)
        istream & in;       // the underlying byte-stream

    public:
        BitInputStream(istream & is) : in(is), buf(0), nbits(8) {}

        void fill() {
            buf = in.get(); // Pull 1 byte from the byte-buffer
            nbits = 0;      
        }

        unsigned int readBit() {
            if(nbits == 8) fill();

            // Logic to extract the specific bit from the buffer
            // Example: unsigned int bit = (buf >> (7 - nbits)) & 1;
            nbits++;
            return bit;
        }
};
```

---
## Efficiency Considerations

In the `BitOutputStream::flush()` function, we could call `out.flush()` to force data to the disk immediately. However, this is **optional and slow**. It defeats the purpose of the bytewise buffer by forcing a disk write every time a single byte is completed. To keep things fast, only flush the bitwise buffer frequently; let the operating system handle the larger bytewise-to-disk flush.