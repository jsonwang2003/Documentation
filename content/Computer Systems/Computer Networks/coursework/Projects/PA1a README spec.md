# Design Document
## Frame Format
- `remaining_msg_bytes`(`uint16_t`)
- `dst_id`(`uint8_t`)
- `src_id`(`uint8_t`)
- `seq_num` (`uint8_t`)
- `typeof_frame` (`uint8_t`): 0 for data frame, 1 for ack frame
- `data` (`char [57]`)
- `trailer` (`uint8_t`): CRC-8 checksum field
Total bytes in frame: 64 bytes
## Key Design Decisions
- For error detection, any frame's outgoing will compute the crc8 using the provided function and store the remainder in `trailer` field of the frame, the receiving side will then do compute crc8 and any remainder that is non-zero indicates frame has error and ignore the frame as non-existent.
- The ack frame's `seq_num` is the next expected frame the receiver expects, implying that the sender should treat the difference between the base `seq_num` and ack's `seq_num` to be the number of frames the sender should slide their window
- The `seq_num` is maintained per sender-receiver pair and continues across multiple messages. This design choice implies the sender maintain a list of heads of the `seq_num` for each destination host and the receiver maintains a receiver window for each sender to maintain the frame sequence. The continuation of `seq_num` across multiple messages across the same connection avoids the edge case that if the last ACK is lost, the retransmitted frame looks identical to the start of the next message unless numbering continues.
- The `seq_num` wraps around implicitly by having the data type of `uint8_t` where at `seq_num` 255 + 1 will store back to 0 and calculate the difference using `seq_num_diff(x1, x2)` which computes the signed distance between two sequence numbers correctly across the range of 255 to 0 wrap by giving back the result of the relative distance the `x2` is from `x1`, positive meaning after the `x1` and negative meaning before `x1`.
- Maintain the sliding window in sender by running a for-loop through the `send_window` and copies the content of the window slot. The acknowledged frames and their timeouts are freed before the remaining slots shift down. Slot 0 is always the oldest unacknowledged slot and shifts the rest down.
- The receive window stores `num_hosts` windows for each connections. Each receive window slot stores the next expected seq num, sliding window of frame pointers, message buffer, and a message length counter. Maintain `num_hosts` receive windows for each connection by running a for-loop though the `receive_window` and copies the content of the window slot using similar idea of `send_window`. The delivered frames are freed, the rest of the frames shifting down, and `NULL` the remaining frames. Slot 0 is always the next expected frame and shifts the rest down.
- Suppose a duplicate ack received by the sender host, it points at or before the oldest frame that has yet to receive an ack, so the difference is 0 or negative and the ack is ignored. This is mainly due to any prior frames that has received acks has already been freed in memory.
- Suppose a duplicate frame received by the receiver, it points before the next expected frame, which is not delivered again, but is still acknowledged right away with the next expected frame's `seq_num` so the sender stops retransmitting the duplicate frame 
- For timed out frames in sender, the timeout in an occupied window slot is set to `NULL` to indicate the frame has timed out, which the resend will give back the fresh timeout.
- Suppose a message exceeds `FRAME_PAYLOAD_SIZE = 57` bytes, message is fragmented by copying raw byte data into the frame where the receiver retrieves the 57 bytes without the null terminator, which receiver then concatenate the data onto the message buffer with null terminator at the end. The field `remaining_msg_bytes` represents how much more bytes does the message have remaining, which is calculated by the remaining message length subtracted by the minimum value between remaining message length and payload size, giving the field `remaining_msg_bytes` a range from 0 up to one less than the message length.

## Major Functions
- `host_get_next_expiring_timeval(host)`: Find the earliest time in send window's timeouts and return said time
- `handle_incoming_acks(host, curr_timeval)`: Ignores any data frames and validates any ack frames and slides the send window forward
- `handle_input_cmds(host, curr_timeval)`: Turn the terminal commands into frames, assign sequence numbers, compute CRC8, and store in send buffer.
- `handle_timedout_frames(host, curr_timeval)`: Mark all frames in the window by freeing the old timeout and setting time to `NULL` if that frame has a timeout time earlier than the current time. 
- `handle_outgoing_frames(host, curr_timeval)`: Sends a copy of each timed-out frame and each new frame, keeps the original in the window, and gives each a deadline of 100ms ahead (with 10ms of delay between the frames)
- `handle_incoming_frames(host)`: For all frames coming through the incoming frames queue, check and mark off the receiver window, drop any corrupted frames, reassemble back the fragmented message, print once no bytes in message remaining, and send the ack frame with `seq_num` of the next expected sequence number back to the sender host.