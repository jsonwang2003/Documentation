>[!INFO]
>A way of encoding **codepoint numbers** that will correspond to a specific character or symbol (including ASCII)


- **Backwards compatible** with [[ASCII]]
  → The encoding for **1 byte** is the same for the encoding for **ASCII**

| 1 Byte | 0xxxxxxx |          |          |          | (ASCII) |
| ------ | -------- | -------- | -------- | -------- | ------- |
| 2 Byte | 110xxxxx | 10xxxxxx |          |          |         |
| 3 Byte | 1110xxxx | 10xxxxxx | 10xxxxxx |          |         |
| 4 Byte | 11110xxx | 10xxxxxx | 10xxxxxx | 10xxxxxx |         |
