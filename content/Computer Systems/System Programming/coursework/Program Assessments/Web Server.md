## Web Servers and HTTP

[HTTP](https://en.wikipedia.org/wiki/HTTP) is one of the most common protocols for communicating across computers. At the systems programming level, this means using system calls (usually in C) to tell the operating system to send bytes over a network.

One nice feature of HTTP is that it is a primarily text-based protocol, which makes it more straightforward for humans to read and debug. It is also well-understood by web browsers and programs like `curl`, making it easy to test and connect to user-facing devices.

It's useful to get experience with the format of HTTP, and with using system calls in C to manipulate HTTP requests.

## Task – Chat Server

In this programming assignment, you'll write a C program to implement a chat room (think a plain-text version of [Slack](https://slack.com/) or [Discord](https://discord.com/)).

It's best to complete the PA on `ieng6`, because it gives a consistent testing environment for the live server.

Your programs should compile and run with:

`make chat-server ./chat-server <optional port number>`

The server should start with ./chat-server and print a single message:

`$./chat-server Server started on port PPPPP`

If a port number was provided, it should use that port, otherwise it should print an open port that was selected.

It should continue running, listening for requests on that port, until shutdown with Ctrl-c. It can print any other logging messages or other output needed to the terminal.

The requests the chat server listens for are described in this section:

### /chats

A request to `/chats` responds with the plain text rendering of all the chats.

The rendered chat format is

```c
[#N 20XX-MM-DD HH:MM] <username>: <message> (<rusername>) <reaction> 
... [more reactions] ... 
... [more chats] ...
```

Chats must be rendered properly, as in HW5.5. You can put in whatever effort you like into lining things up nicely within these constraints, but these are the requirements.

Example chats rendering might look like:

```txt
[#1 2025-11-06 09:01]         joe: hi aaron 
[#2 2025-11-06 09:02]       aaron: sup 
[#3 2025-11-06 09:04]         joe: working on the example chat for the PA 
[#4 2025-11-06 09:06]       aaron: oh cool what should it say 
[#5 2025-11-06 09:07]         joe: I dunno we could go pretty meta with it? like a chat about the chat 
[#6 2025-11-06 09:10]       aaron: eh kinda lame tbh 
[#7 2025-11-06 09:11]         joe: whatever I already wrote it, going with it as-is 
[#8 2025-11-06 09:12]       aaron: ok but make sure we don't look like jerks 
[#9 2025-11-06 09:12]       aaron: or at least not me                             
							(joe)  👍🏻 
[#10 2025-11-06 09:12]         joe: good talk
```

### /post

A `post` request looks like this:

`/post?user=<username>&message=<message>`

This creates a new chat with the given username and message string with a timestamp given by the time the request is received by the server. It must respond with the list of all chats (including the new one).

Limits and constraints:
- If a parameter (username or message) is missing, respond with some kind of error (HTTP code 400 or 500)
- If username is longer than 15 bytes, respond with some kind of error (HTTP code 400 or 500)
- If message is longer than 255 bytes, respond with some kind of error (HTTP code 400 or 500)
- If a post would make there be more than 100000 (one hundred thousand) chats, the server should respond with an error (HTTP code 404 or 500)

### /react

`/react?user=<username>&message=<reaction>&id=<id>`

Creates a _new reaction_ to a chat by the given username with the given message string, reacting to the post with the given id (the ids are the `#N` at the beginning of posts). It must respond with the list of all chats (including the new one).

Limits and constraints:
- If the id is not the ID of some existing chat, respond with some kind of error (HTTP code 400 or 500).
- If a parameter (username or message or id) is missing, respond with some kind of error (HTTP code 400 or 500)
- If username is longer than 15 bytes, respond with some kind of error (HTTP code 400 or 500)
- If message is longer than 15 bytes, respond with some kind of error (HTTP code 400 or 500) – reactions are intended to be short!
- If a reaction would make a chat have more than 100 reactions, the server should respond with an error (HTTP code 404 or 500)

## **Implementation Guide**

This page is the entire _specification_ for the assignment; it's what you need to implement. You are free to make whatever choices you like in your code within these constraints. To help you on your way, we have some _implementation notes_ below with suggestions and ideas for how to get started and what to think about. These are not requirements, just suggestions!

First, make sure to do **[[Computer Systems/System Programming/coursework/Problem Sets 5/index|Problem Sets 5]]** first if you haven't already! While completing it you will create helper functions you can use as well as help you develop an understanding of your task.

Then, one way to break down the work the server needs to do is:
- Parsing and interpreting requests (is the request a new post, a reaction, etc)
- Updating the current data (chats and reactions) based on the parameters in the request
- Responding to requests based on the current state of the data (chats and requests)

One way to work incrementally is to separate the _data handling_ and the _request handling_ parts into different functions.

The _data handling_ functions can be tested with by writing a `main` and using `printf` or `assert`, and the _request handling_ can be tested with `curl` or a client.

We think the following functions might be useful for you to implement. In your program you might have slightly different signatures or ideas, but these are a useful starting point. Also, our staff is more familiar with this approach, so it will take us less time to help you in office hours!

### Data Handling Functions

These functions can be written and tested without starting a server at all. You could consider having a separate `main` function in its own file that just tests these!

#### `add_chat`
A function `add_chat` can add a single chat.

```c
uint8_t add_chat(char* username, char* message)
```

This function might have several tasks:
1. Update the current `id`
2. Get the current timestamp
3. Create a new chat and fill in its username and message fields
4. As needed, allocate new space, put the new chat in heap memory, store a reference to it in an array, etc. depending on your specific representation of chats

This is testable by setting up initial states of chats and reactions, running the function, and then using `assert` or `printf` on the results.

#### `add_reaction`
Similar to `add_chat`, this adds a single reaction:

```c
uint8_t add_reaction(char* username, char* message, char* id)
```

This function might have several tasks:
- Use the id to locate the chat that this reaction is for (and maybe return early with an error if the id is invalid/out of range)
- Create a new reaction and fill in its username and message fields
- Add the reaction to the chat struct somehow, maybe with newly allocated space, an added element or reference in an array, etc. depending on your specific representation of chats and reactions
- Update the count of reactions on the referenced chat

This is testable by setting up initial states of chats and reactions, running the function, and then using `assert` or `printf` on the results.

### Request and Response Handling Functions

You will definitely need to write a function for handling responses. But the work of handling individual responses can be broken up. One approach could be to get the path and query parameter string from the request and check if it's path is `/post`, `/chats`, etc, then pass the string to other functions

#### `respond_with_chats`

```c
void respond_with_chats(int client)
```

This function is responsible for using `write` or `send` to send the response to the client that made the request. It might include:
- Using `snprintf` to format strings with data from the timestamp or ids
- Calling `write(client, str, size)` on various strings (with the appropriate size) to directly send the data to the client

#### `handle_post`

```c
// path is a string like "/post?user=joe&message=hi"
void handle_post(char* path, int client)
```

This function can have several tasks:
- Use string functions to extract the username and message from the path
- Call `add_chat` to do the data update
- Call `respond_with_chats` to send the response

#### `handle_reaction`

```c
// path is a string like "/react?user=joe&message=hi&id=3"
void handle_reaction(char* path, int client)
```

This function can have several tasks:
- Use string functions to extract the username, message, and id from the path
- Call `add_reaction` to do the data update
- Call `respond_with_chats` to send the response

## Representing Chats and Reactions
Chats and reactions both have multiple fields, so a natural choice is to represent both chats and reactions as structs.

A chat has several components, which may be good candidates for struct fields:
- The message
- The username
- The timestamp
- The reactions to the message

A reaction has the message content and the user who posted it (no timestamp or reactions-to-reactions), both of which are fixed-size.

You should make `struct`s to hold the `Chat`s and `Reaction`s in your server, and use the constrains above about the lengths of usernames, messages, and so on to help you decide what data to store.

## Other Helpful Functions

This PA explores several features that are straightforward to use, but there are _many_ of them. We might add more to this list as the PA goes on! Here are a few functions you'll probably find useful; try `man` on them, or follow the links, or do your own searching and research. Don't forget all the functions from class (e.g. `malloc` and other allocation functions, `strstr` and other string manipulation functions, and so on). This list is mainly focused on things we haven't tried in class.
- [`atoi`](https://cplusplus.com/reference/cstdlib/atoi/?kw=atoi): convert `char*` to integer
- **Time functions:**
    - [`time`](https://en.cppreference.com/w/c/chrono/time): get the current time
    - [`localtime`](https://en.cppreference.com/w/c/chrono/localtime): convert the time to the current local time zone
    - [`strftime`](https://en.cppreference.com/w/c/chrono/strftime): print the time in a given format

---
## Code
Repository found [here](https://github.com/ucsd-cse29-fa25/pa5-web-server-jsonwang2003)

```c
#include "http-server.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

char header[] = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n";
uint32_t chat_count = 0;
uint32_t global_id = 0;

typedef struct Reaction{
	char username[16];
	char message[256];
} Reaction;

typedef struct Chat{
	uint32_t chat_id;
	char message[256];
	char username[16];
	time_t timestamp;
	size_t reaction_count;
	Reaction reactions[100];
} Chat;

Chat* chats[100000];

void send_str(int client, const char* s){
	size_t len = strlen(s);
	send(client, s, len, 0);
}

void http_error(int client, int code, const char* msg){
	char buffer[256];
	snprintf(buffer, sizeof(buffer), "HTTP/1.1 %d Error\r\nContent-Type: text/plain\r\n\r\n%s\n", code, msg);
	send_str(client, buffer);
}

void url_decode(char* s){
	char* src = s;
	char* dst = s;
	while(*src){
		if(*src == '+'){
			*dst++ = ' ';
			src++;
		} else if(*src == '%' && src[1] && src[2]){
			char hex[3] = { src[1], src[2], '\0'};
			*dst++ = (char)strtol(hex, NULL, 16);
			src+=3;
		} else {
			*dst++ = *src++;
		}
	}
	*dst = 0;
}

uint8_t add_chat(char* username, char* message){
	if(!username || !message) return 1;
	size_t ulen = strlen(username);
	size_t mlen = strlen(message);
	if(ulen == 0 || ulen > 15) return 2;
	if(mlen == 0 || mlen > 255) return 3;
	
	Chat* c = (Chat*)malloc(sizeof(Chat));
	if(!c) return 4;

	c->chat_id = ++global_id;
	strncpy(c->username, username, sizeof(c->username) - 1);
	c->username[sizeof(c->username) - 1] = 0;
	strncpy(c->message, message, sizeof(c->message) - 1);
	c->message[sizeof(c->message) - 1] = 0;
	c->timestamp = time(NULL);
	c->reaction_count = 0;

	chats[chat_count++] = c;
	return 0;
}

uint8_t add_reaction(char* username, char* message, uint32_t id){
	if(id == 0 || id > global_id) return 1;

	Chat* target = NULL;
	for(size_t i = 0; i < chat_count; i++){
		if(chats[i]->chat_id == id){
			target = chats[i];
			break;
		}
	}
	if(!target) return 2;
	if(target->reaction_count >= 100) return 3;

	Reaction* r = &target->reactions[target->reaction_count++];
	strncpy(r->username, username, sizeof(r->username) - 1);
	r->username[sizeof(r->username) - 1] = 0;
	strncpy(r->message, message, sizeof(r->message) - 1);
	r->message[sizeof(r->message) - 1] = 0;

	return 0;
}

void respond_with_chats(int client){
	send_str(client, header);

	char line[1024];
	char formatted_time[32];

	for(size_t i = 0; i < chat_count; i++){
		Chat* c = chats[i];
		struct tm* timeinfo = localtime(&c->timestamp);
		strftime(formatted_time, sizeof(formatted_time), "%Y-%m-%d %H:%M:%S", timeinfo);
		snprintf(line, sizeof(line), "[#%u %s] %12s: %s\n", c->chat_id, formatted_time, c->username, c->message);
		send_str(client, line);

		for(size_t r = 0; r < c->reaction_count; r++){
			Reaction* reacts = &c->reactions[r];
			snprintf(line, sizeof(line), "%30s(%s) %s\n", "", reacts->username, reacts->message);
			send_str(client, line);
		}
	}
}

void handle_post(char* path, int client){
	char user[16];
	char msg[256];

	char* ustart = strstr(path, "user=");
	char* mstart = strstr(path, "message=");

	if(!ustart || !mstart){
		http_error(client, 400, "Missing parameters");
		return ;
	}

	ustart += 5;
	char* uend = strpbrk(ustart, "& \r\n");
	size_t ulen = uend ? (size_t)(uend - ustart) : strlen(ustart);
	if(ulen >= sizeof(user)){
		http_error(client, 400, "Username too long");
		return;
	}
	strncpy(user, ustart, ulen);
	user[ulen] = 0;
	url_decode(user);

	mstart += 8;
	char* mend = strpbrk(mstart, "& \r\n");
	size_t mlen = mend ? (size_t)(mend - mstart) : strlen(mstart);
	if(mlen >= sizeof(msg)){
		http_error(client, 400, "Message too long");
		return;
	}
	strncpy(msg, mstart, mlen);
	msg[mlen] = 0;
	url_decode(msg);

	uint8_t rc = add_chat(user, msg);
	if(rc == 0){
		respond_with_chats(client);
	} else {
		http_error(client, 400, "Invalid parameters");
	}
}

void handle_reaction(char* path, int client){
	char user[16];
	char msg[16];
	char id_buffer[32];

	char* ustart = strstr(path, "user=");
	char* mstart = strstr(path, "message=");
	char* idstart = strstr(path, "id=");

	if(!ustart || !mstart || !idstart){
		http_error(client, 400, "Missing parameters");
		return;
	}

	ustart += 5;
	char* uend = strpbrk(ustart, "& \r\n");
	size_t ulen = uend ? (size_t)(uend - ustart) : strlen(ustart);
	if(ulen >= sizeof(user)){
		http_error(client, 400, "Username too long");
		return;
	}
	strncpy(user, ustart, ulen);
	user[ulen] = 0;
	url_decode(user);

	mstart += 8;
	char* mend = strpbrk(mstart, "& \r\n");
	size_t mlen = mend ? (size_t)(mend - mstart) : strlen(mstart);
	if(mlen >= sizeof(msg)){
		http_error(client, 400, "Message too long");
		return ;
	}
	strncpy(msg, mstart, mlen);
	msg[mlen] = 0;
	url_decode(msg);

	idstart += 3;
	char* idend = strpbrk(idstart, "& \r\n");
	size_t idlen = idend ? (size_t)(idend - idstart) : strlen(idstart);
	if(idlen >= sizeof(id_buffer)) idlen = sizeof(id_buffer) - 1;
	strncpy(id_buffer, idstart, idlen);
	id_buffer[idlen] = 0;

	uint32_t id = (uint32_t)atoi(id_buffer);

	uint8_t rc = add_reaction(user, msg, id);
	if(rc == 0){
		respond_with_chats(client);
	} else {
		http_error(client, 400, "Invalid reaction parameters");
	}
}

char* get_message(char* request_path){
	char* message = strstr(request_path, "message=");
	if(message != NULL){
		message += 8;
		char* end = strstr(message, "&");
		if(end != NULL){
			*end = 0;
		}
		return message;
	}
	return NULL;
}

char* get_user(char* post_request){
	char* user = strstr(post_request, "user=") + 5;
	char* end = strstr(user, "&");
	*end = 0;
	return user;
}

int get_port(char* request){
	char* start = strstr(request, "Host: ") + 6;
	start = strstr(start, ":") + 1;

	char* end = strstr(start, "\n");
	size_t len = end - start;

	char port[len + 1];
	strncpy(port, start, len);

	return atoi(port);
}

void simple_handler(char* request, int response_socket){
	int port = get_port(request);
	printf("Server started on port %d\n", port);

	char chats[] = "GET /chats";
	char post[] = "GET /post";
	char react[] = "GET /react";

	if(strncmp(chats, request, strlen(chats)) == 0){
		respond_with_chats(response_socket);
	} else if(strncmp(post, request, strlen(post)) == 0){
		handle_post(strstr(request, "?") + 1, response_socket);
	} else if(strncmp(react, request, strlen(react)) == 0){
		handle_reaction(strstr(request, "?") + 1, response_socket);
	} else {
	    char* error = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\nNot found";
		send(response_socket, error, strlen(error), 0);
	}
}

int main(int argc, char** argv){
	if(argc == 2){
		start_server(&simple_handler, atoi(argv[1]));
	} else {
		start_server(&simple_handler, 0);
	}
}
```