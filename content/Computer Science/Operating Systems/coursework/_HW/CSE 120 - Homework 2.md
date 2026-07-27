**Due:** Thursday, April 24 at 11:59pm

For the homework questions below, if you believe that you cannot answer a question without making some assumptions, state those assumptions in your answer.

---
# Question 1

One of the goals of this question is to give you practice with context switching and thread queue manipulation in Nachos, and the hope is that you will find it useful for working on project 1 (so there is value in doing this problem well before the due date).

Consider the following test program for an implementation of `KThread.join` in Nachos. It begins when the main Nachos thread calls `KThread.selfTest`. You do not need to know the details of how join is implemented. All you need to know is that when a parent thread calls join on a child thread, the parent does one of two things: (1) if the child is still running, the parent blocks until the child finishes (at which point the parent is placed on the ready queue); (2) if the child has finished, the parent continues to execute without blocking. Assume join uses a wait queue of some kind in its implementation.

```java
private static class A implements Runnable {
    A () {}
    public void run () {
        KThread t2 = new KThread (new B()).setName ("B");
        System.out.println ("foo");
        t2.fork();
        System.out.println ("far");
        t2.join();
        System.out.println ("fum");
    }
}

private static class B implements Runnable {
    B() {}
    public void run () {
        System.out.println ("fie");
    }
}

public static void selfTest() {
    KThread t1 = new KThread (new A()).setName ("A");
    System.out.println ("fee");
    t1.fork();
    System.out.println ("foe");
    t1.join();
    System.out.println ("fun");
}
```

Assume that the scheduler runs threads in FIFO order with non-preemptive scheduling (no preemptive time-slicing) on a single CPU core, and threads are placed on wait queues in FIFO order. Trace the execution of this program until it returns from `selfTest` and (a) write the sequence of context switches that occurred up this point, (b) write the output of the program, and (c) list the queues that the threads are on, and their relative order if more than one thread is on a queue.

**a.** Context switches: 
	main → A → B → A → main
**b.** Output: 

```txt
fee
foe
foo
far
fie
fum
fun
```

**c.** Thread queues when `selfTest` returns:

| Step               | currentThread | readyQueue  | join wait Queue |
| ------------------ | ------------- | ----------- | --------------- |
| start              | main          | $\emptyset$ | $\emptyset$     |
| before `t1.join()` | main          | A           | $\emptyset$     |
| before `t2.join()` | A             | B           | main            |
| after `t2.join()`  | B             | $\emptyset$ | main, A         |
| B finishes         | A             | $\emptyset$ | main            |
| A finishes         | main          | $\emptyset$ | $\emptyset$     |


> **Hint:** First try the problem by following the code manually, keeping track of which queues threads are on using paper. Then, once you have implemented join, try adding the code as a test in `KThread.java` and running it to check your answer.

---
# Question 2

The Intel x86 instruction set architecture provides an atomic instruction called `XCHG` for implementing synchronization primitives. (If you are curious, this reference page shows the full syntax and semantics of the instruction.) Semantically, `XCHG` works as follows (although keep in mind that it is executed atomically):

```c
void XCHG (bool *X, bool *Y) {
    bool tmp = *X;
    *X = *Y;
    *Y = tmp;
}
```

**a.** Show how `XCHG` can be used instead of test-and-set to implement the `acquire()` and `release()` functions of the spinlock data structure described in the "Synchronization" lecture.

```c
struct lock {
	bool held = False;
}

void acquire (struct lock *) {	
	bool guardValue = True;
	while (guardValue){
		XCHG(&lock->held, &guardValue);
	}
}

void release (struct lock *) {
	bool free = False;
	XCHG(&lock->held, &free);
}
```

**b.** Briefly explain why your implementation guarantees mutual exclusion. Why is atomicity of `XCHG` essential to the correctness of `acquire()`? 

Mutual exclusion is guaranteed because the `XCHG` ensures that only 1 thread can successfully swap a `False` value out of `lock->held` and replace it with `true` at any given time. Atomicity of `XCHG` is essential to the correctness of `acquire()` because when checking the current state must not be interrupted by any context switch. 

**c.** What are the tradeoffs between using a spinlock (as implemented above) versus a blocking lock (one that puts the waiting thread to sleep)? Under what circumstances would you prefer one over the other?

When using a spinlock, since thread does not go to sleep, there is no additional resources being consumed when context switching or thread rescheduling. But spinlock can be wasteful of CPU cycles. Prefer to use when **critical section is short** where the clock cycles spent waiting is less than waking up the thread.

On the other hand, when using a blocking lock, since the thread is being put to sleep, it gives up the CPU so other threads can perform work. But waking the thread up later requires additional resources (getting back to progress, the old values in registers, etc.). Prefer to use when **critical section is long** where the clock cycles spent waking up the thread is worth it compared to just waiting for the thread in spinlock. 

---
# Question 3

A common pattern in parallel scientific programs is to have a set of threads do a computation in a sequence of phases. In each phase $i$, all threads must finish phase $i$ before any thread starts computing phase $i+1$. One way to accomplish this is with barrier synchronization. At the end of each phase, each thread executes `Barrier.done(n)`, where `n` is the number of threads in the computation. A call to `Barrier.done` blocks until all of the `n` threads have called `Barrier.done`. Then, all threads proceed. You may assume that the process allocates a new `Barrier` for each iteration, and that all threads of the program will call `done` with the same value.

**a.** Use pseudocode to implement a classic monitor that implements `Barrier`.

```Java
monitor Barrier {
	private int count = 0;
    private Condition allArrived;
    
    void done (int n) {
	    count++;
	    
	    if (count < n){
		    allArrived.wait();
	    } else {
		    count = 0;
		    allArrived.broadcast();
	    }
    }
}
```

**b.** Use pseudocode to implement `Barrier` using an explicit lock and condition variable (Mesa semantics, as implemented in Project 1).

```Java
class Barrier {
    private int count = 0;
    private Lock lock = new Lock();
    private Condition allArrived = new Condition(lock);
    
    void done(int n){
	    lock.acquire();
	    
	    count++;
	    
	    if (count < n){
			allArrived.wait();
	    } else {
			count = 0;
			allArrived.broadcast();
	    }
	    
	    lock.release();
    }
}
```

**c.** Use pseudocode to implement `Barrier` using an explicit lock and one semaphore.

> **Hint:** think about how you can make sure that all $n-1$ waiting threads are woken up. There are multiple possible approaches.

```Java
class Barrier {
    private int count = 0;
    private Lock lock = new Lock();
    private Semaphore sem = new Semaphore(0);
    
    void done(int n) {
	    lock.acquire();
	    
	    count++;
	    
	    if(count < n){
			lock.release();
			sem.wait();
	    } else {
		    count = 0;
		    
		    for (int i = 0; i < n-1; i++){
			    sem.signal();
		    }
		    
		    lock.release();
	    }
    }
}
```

---
# Question 4

Torrey Pines would like your help synchronizing surfers and the ocean. Using pseudocode, implement the class `Surfing` using locks and condition variables to synchronize multiple surfer threads with one ocean thread (do not manipulate interrupts). Your solution also cannot change the lock, condition variable, or thread classes, and do not use data structures other than locks and condition variables to store references to threads.

You need only use pseudocode in your answers. Your pseudocode does not have to compile, and you can use whatever syntax you are most comfortable with (the solutions use the Nachos syntax). But it does have to look like code.

The `Surfing` class can be in one of two states, either breaking or calm. Surfer threads invoke the `paddle` method to indicate the direction, left or right, they would like to surf the break. When calm, surfer threads block until the next wave arrives. When breaking, surfer threads block if the wave is not breaking in their direction, and otherwise return immediately.

The ocean thread invokes the `wave` method indicating the direction the next wave is breaking (left, right, or both ways). It changes the state to breaking and wakes up all surfer threads waiting to catch waves in that direction, or wakes up all threads if the wave is breaking in both directions. It invokes the `done` method to indicate that the wave is finished breaking, changing the state back to calm. The ocean thread alternates invoking `wave` and `done`, and the state is initially calm.

```java
class Surfing {
    enum State { CALM, BREAKING; }
    enum Direction { LEFT, RIGHT, BOTH; }
    
    private State currentState;
    private Direction currentWaveDirection;
    private Lock lock;
    private Condition waveCondition;

    Surfing () {
	    this.currentState = State.CALM;
	    this.lock = new Lock();
	    this.waveCondition = new Condition(lock);
    }

    void paddle (Direction dir) {
        // invoked by surfer threads
        lock.acquire();
        
        // if state is calm or wave is breaking but not in their direction
        while(currentState == State.CALM || 
		        (currentWaveDir != dir && currentWaveDir != Direction.BOTH)) {
			waveCondition.sleep();
	    }
	    
	    lock.release();
    }

    void wave (Direction dir) {
        // invoked by the ocean thread
        lock.acquire();
        
        this.currentState = State.BREAKING;
        this.currentWaveDir = dir;
        
        waveCondition.wakeAll();
        
        lock.release();
    }

    void done () {
        // invoked by the ocean thread
        lock.acquire();
        
        this.currentState = State.CALM;
        
        lock.release();
    }
}
```

---
# Question 5

Eleanor, Chidi, Tahani, and Jason are working on their term papers in CSE 120, which is a 10,000 word essay on My All-Time Favorite Race Conditions. To help them work on their papers, they have one dictionary, two copies of Roget's Thesaurus, and two coffee cups.

- Eleanor needs to use the **dictionary** and a **thesaurus** to write her paper;
- Chidi needs a **thesaurus** and a **coffee cup** to write his paper;
- Tahani needs a **dictionary** and a **thesaurus** to write her paper;
- Jason needs **two coffee cups** to write his paper (he likes to have a cup of regular and a cup of decaf at the same time to keep himself in balance).

Consider the following state:

- Eleanor has a **thesaurus** and needs the *dictionary*.
- Chidi has a **thesaurus** and **a coffee cup**.
- Tahani has the **dictionary** and needs a *thesaurus*.
- Jason has **a coffee cup** and needs *another coffee cup*.

Is the system deadlocked in this state? Explain using a resource allocation graph as a reference.

![[Pasted image 20260423114745.png]]

Since Chidi has both items required to finish her paper, she will eventually give out a thesaurus and a coffee cup. After the release, Jason will be finishing his paper as there are now 2 cups being free to use. At the same time, Tahani will be finishing his paper too as he now holds the thesaurus Chidi had, which will eventually finish his paper to give back the items. After Tahani finishes, Eleanor will finish as Tahani released the dictionary needed for Eleanor to finish his paper. Therefore, there is no deadlock in this state.