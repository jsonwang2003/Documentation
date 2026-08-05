The baseline Nachos implementation has an incomplete thread system. In this project, your job is to complete it, and then use it to provide more elaborate synchronization functionality.

## Background

Properly synchronized code should work no matter what order the scheduler chooses to run the threads on the ready list. In other words, we should be able to put a call to `KThread.yield` (causing the scheduler to choose another thread to run) anywhere in your code where interrupts are enabled, and your code should still be correct. Later assignments will require properly synchronized code, so these topics are important going forward.

To induce context switches, Nachos will cause `KThread.yield` to be called on your behalf in a repeatable (but sometimes unpredictable) way. Nachos code is repeatable in that if you call it repeatedly with the same arguments, it will do exactly the same thing each time. However, if you invoke `nachos -s <number>` with a different number each time, calls to `KThread.yield` will be inserted at different places in the code (see the Testing section below).

You will be modifying source code files in the threads subdirectory (e.g., `threads/Alarm.java`), and compiling in the `proj1` subdirectory. As described on the main page, be careful to only modify and add classes to existing source files (do not create new source files). There should be no busy-waiting in any of your solutions to this assignment. (The initial implementation of `Alarm.waitUntil` is an example of busy-waiting.)
## Tasks
### Task 0
Browse through the initial thread system implementation, starting with `KThread.java`. This thread system implements **thread fork**, **thread completion**, and **semaphores** for synchronization. It also provides **locks** and **condition variables** built on top of semaphores.

Trace the execution path (by hand) for the startup test case provided. When you trace the execution path, it is helpful to keep track of the state of each thread and which procedures are on each thread's execution stack. You will notice that when one thread calls `TCB.contextSwitch`, that thread stops executing, and another thread starts running. The first thing the new thread does is to return from `TCB.contextSwitch`. We realize this will seem cryptic to you at first, but you will understand threads once you understand why the `TCB.contextSwitch` that gets called is different from the `TCB.contextSwitch` that returns.

Prep for CSE 120 as you did for task 1 in project 0, then compile and run the baseline implementation of Nachos in the proj1 directory: 
```bash
% cd nachos/proj1
% make
% nachos
```
  

### Task 1
Complete the implementation of the `Alarm` class (except for `cancel`, which you will implement later) . A thread calls `waitUntil(long x)` to suspend its execution until wall-clock time has advanced to at least $\text{now} + x$. This method is useful for threads that **operate in real time**, such as 
- clock application that wakes up once a second

There is no requirement that threads start running immediately after waking up; just **put them on the ready queue in the timer interrupt handler** after they have waited for at least the right amount of time. Do not fork any additional threads to implement `waitUntil`; you need to modify `waitUntil` and the timer interrupt handler methods.

Feel free to add additional data structures or classes to the existing source files but do not create new source files.  `waitUntil` itself, though, is not limited to being called by one thread; any number of threads may call it and be suspended at any one time. If the wait parameter $x$ is $0$ or **negative**, return **without waiting** (*do not assert*). Also be careful if you are deleting while iterating (a common general programming issue).

Note that only one instance of `Alarm` may exist at a time, and Nachos already creates one global alarm that is referenced via `ThreadedKernel.alarm`.

#### Testing: 
For examples and strategies for implementing tests, see the [[#Testing|Testing Section]] (note that if you fork new threads, you need to make sure that the main thread does not terminate before its children). Implement tests that verify that a thread waits (approximately) for its requested duration; if the wait parameter is $0$ or **negative**, the thread does not wait; multiple threads waiting on the alarm are woken up at the proper times, and in the proper time order.

### Task 2
Implement join functionality by modifying `KThread.join` and `KThread.finish`. `KThread.join` synchronizes the *calling* thread with the completion of the *called* thread. As an example, if thread **B** executes the following:
```java
KThread A = new KThread (...);
...
A.join ();
```

we say that thread **B** joins with thread **A**. When **B** calls `A.join`, there are two possibilities. 
1. If **A** has already finished, then **B** returns immediately from join without waiting. 
2. If **A** has not finished, then **B** waits inside of join until **A** finishes (it should not busy wait) → when **A** finishes, it resumes **B**. 

Often thread **B** is called the "parent" and **A** is called the "child" since a common pattern is for a thread that creates child threads to join on them to wait for them to finish. However, note that any thread can call `KThread.join` on another (it does not have to be a parent/child relationship).

#### Note that: 
- `join` does not have to be called on a thread, a thread can finish successfully even if no other thread calls join on it.
- A thread cannot join to itself (the initial implementation already checks for this case and invokes `Lib.assert` when it happens, keep this `Lib.assert` call in your code)
- `join` can be called on a thread at most once—if thread **B** calls join on **A**, then it is an **assert error** for **B** or any other thread **C** to call join on **A** again.
#### Testing: 
Implement tests that verify 
- if **B** calls `join` on **A** and **A** is still executing, **B** waits; if **B** calls `join` on **A** and **A** has finished executing, then **B** does not wait
- if a thread calls `join` on **itself**, Nachos asserts
- if `join` is called *more than once on a thread*, Nachos asserts
- one thread can `join` with multiple other threads in succession; independent pairs of threads can join with each other without interference.

### Task 3
Implement condition variables using **interrupt disable** and **restore to provide atomicity**. The class `Condition` is a sample implementation that uses `semaphores`, and your job is to **provide an equivalent implementation** in class `Condition2` by *manipulating interrupts* instead of using `semaphores`. Once you are done, you will have two alternative implementations that provide the exact same functionality. Examine the existing implementation of the `Lock` class to guide you on how to manipulate interrupts for when you implement the methods of `Condition2`. For this part, you do not have to implement `sleepFor`.

A thread must have acquired the **lock** associated with the **condition variable** when it invokes methods on the CV. The underlying implementation of the `Lock` class already has code to assert in these cases, but we recommend writing a test program that causes such an error so that you can see what happens.

#### Testing: 
Implement tests that verify that 
- `sleep` blocks the calling thread
- `wake` wakes up at most one thread, even if multiple threads are waiting
- `wakeAll` wakes up all waiting threads
	- if a thread calls any of the synchronization methods without holding the lock, Nachos asserts; 
- `wake` and `wakeAll` with no waiting threads have **no effect**, yet future threads that `sleep` will still block (i.e., the `wake`/`wakeAll` is "lost", which is in contrast to the semantics of semaphores).

### Task 4
Since threads waiting on condition variables **typically do so in a loop**, checking whether the situation they are waiting for is true after waking up, it is usually safe to wake up threads waiting on condition variables at any time. As a result, condition variables implemented in many programming languages also support a "scheduled wait" operation where threads can wait on a condition variable with a timeout. 

Implement the `sleepFor` method of `Condition2` and the `cancel` method of `Alarm` to provide this functionality (modifying other methods as necessary). 
- With `sleepFor(x)`, a thread is woken up and returns either because 
	1. Another thread has called `wake` as with `sleep`
	2. The timeout **x** has expired.

#### Testing: 
Implement tests that verify that a thread that calls `sleepFor` will timeout and return after **x** ticks 
- if no other thread calls `wake` to wake it up
- A thread that calls `sleepFor` will wake up and return if another thread calls `wake` before the timeout expires
- `sleepFor` handles multiple threads correctly (e.g., different timeouts, all are woken up with `wakeAll`).

### Task 5
Implement the `Rendezvous` class to provide a mechanism for **threads to exchange values**, using *locks* and *condition variables* to manage concurrency. In contrast to the previous problems, your solution cannot 
- disable/restore interrupts
- change the lock
- condition variable, or 
- thread classes
And it **cannot** use data structures other than *locks* and *condition variables* to store references to threads (but do use data structures to store references to locks and CVs).

`Rendezvous` provides similar functionality as the [rendezvous system call](http://man.cat-v.org/plan_9/2/rendezvous) from the [Plan 9](https://en.wikipedia.org/wiki/Plan_9_from_Bell_Labs) operating system. The same functionality is also provided by the [Exchanger](https://docs.oracle.com/en/java/javase/13/docs/api/java.base/java/util/concurrent/Exchanger.html) class in Java (but you cannot use Exchanger in your implementation). Quoting from the Plan 9 paper, the system used it as a low-level synchronization primitive "to implement communication channels, queuing locks, multiple reader/writer locks, and the sleep and wakeup mechanism...This primitive is sufficient to implement the full set of synchronization routines."

`Rendezvous` has just one method, `exchange`. 
- The first thread **A** to call `exchange` with value **X** will block waiting for another thread **B**. 
- When thread **B** calls exchange with value **Y**, it will unblock **A** and the threads will exchange values: 
	- value **Y** will be returned to thread **A**, and value **X** will be returned to thread **B**. 
- When **more than two threads are in exchange at the same time**, `exchange` only needs to ensure that 
	- exchanges match two threads together
	- each thread is only involved in one exchange (e.g., if threads **A-D** call exchange, then `exchange` can pair those threads in any combination as long as **each thread only exchanges with one other**). 
	- An "odd" thread will remain in exchange indefinitely waiting for another thread to exchange with.

`exchange` also takes a tag argument. Different integer tags are used as different, parallel synchronization points (i.e., threads synchronizing on different tags do not interact with each other). The same tag can also be used repeatedly for multiple exchanges. Note that there can be different instances of `Rendezvous`, each of which would synchronize threads completely independently of each other.

#### Tip: 
Implement `Rendezvous` in stages. When starting, **ignore tags and assume threads are synchronizing on the same tag**. 
1. Implement synchronizing two threads exchanging values, and test. 
2. Extend your implementation to **many threads exchanging values on the same tag**, and test. 
3. Handle **multiple tags**, and then multiple instances of `Rendezvous`.

#### Testing: 
Implement tests that verify that: 
- A thread only returns from exchange when **another thread synchronizes with it**
- `exchange` returns the exchanged values from the threads properly
- Many threads can call `exchange` on the **same tag**, and `exchange` will correctly pair them up and exchange their values
- Threads exchanging values on **different tags** operate *independently* of each other
- Threads exchanging values on **different instances** of `Rendezvous` operate *independently* of each other.

### EC. 
Web programming proliferated an asynchronous programming style, where performing some action triggers an asynchronous operation that takes an unknown amount of time (e.g., downloading an image from a Web server). An early design pattern for a program to know when the operation finishes is to associate a callback with the operation: when the operation finishes, the callback is invoked.

Modern languages have added features to streamline writing programs using asynchronous operations (e.g., Promises in JavaScript, Futures in Java, C++, and Rust). In this problem, you will implement a simple version of Futures. 

When instantiating a new Future, programs pass in a function to be invoked **asynchronously** 
> for simplicity, assume the function takes no arguments and returns an int. 

At any point, a program may invoke `get` on the `Future` to obtain the return value of the function. 
- If the function has not completed when `get` is **invoked**, then the caller is **blocked**. 
- If the function has completed, then `get` returns the result of the function. 
	> Note that `get` may be called **any number of times** (potentially by multiple threads), and it should **always return the same value**.

Implement the `Future` class using a Nachos `KThread` to execute the function asynchronously. Happily, the Future class now enables us to effectively have `KThreads` return a result.

## Testing

It is your responsibility to implement your own tests to thoroughly exercise your code to ensure that it meets the requirements specified for each part of the project. Testing is an important skill to develop, and the Nachos projects will help you to continue to develop that skill. Add calls to testing code in `ThreadedKernel.selfTest`, and add class-specific code in `selfTest` methods of each class.

As a testing strategy
1. Start with simple tests and then implement more complicated tests. 
	- When something goes wrong with a **simple** test, it is **easier** to pinpoint what aspect of your implementation has a bug. 
	- When something goes wrong with a more **complicated** test, it is **more difficult** to determine where the bug may be unless you can rule out all the causes that your simple tests have shown to already be correct. 
2. Strongly recommend implementing tests as **separate methods**, rather than making changes to just one or a few methods. 
	- Rather than making a change to an existing test to evaluate new functionality, copy the test into a new method and make the change. 
	- Earlier tests are always there in case you need to use them again. 
	- You can comment out calls to previous tests so that you can concentrate on one test at a time.

To help you get jumpstarted on testing, here are a handful of example test programs across the various problems:
> [!info]- Test for `Wait`
>  ```java
> // Add Alarm testing code to the Alarm class
> public static void alarmTest1() {
> 	int durations[] = {1000, 10*1000, 100*1000};
> 	long t0, t1;
> 	for (int d : durations) {
> 		t0 = Machine.timer().getTime();
> 		ThreadedKernel.alarm.waitUntil (d);
> 		t1 = Machine.timer().getTime();
 >		System.out.println ("alarmTest1: waited for " + (t1 - t0) + " ticks");
> 	}
> }
> 
> // Implement more test methods here ...
> 
> // Invoke Alarm.selfTest() from ThreadedKernel.selfTest()
> public static void selfTest() {
> 	alarmTest1();
> 	// Invoke your other test methods here ...
> }
> ```

> [!Info]- Test for `Join`
> ```java
> // Place Join test code in the KThread class and invoke test methods from KThread.selfTest()
> // Simple test for situation where the child finishes before the parent calls join on it
> 
> private static void joinTest1(){
> 	KThread child1 = new KThread(new Runnable(){
> 		public void run(){
> 			System.out.println("I (heart) Nachos!");	
> 		}	
> 	});
> 	child1.setName("child1").fork();
> 
> 	// We want the child to finish before we call join. Although our solutions to the problems cannot busy wait, our test programs can!
> 	
> 	for(int i = 0; i < 5; i++){
> 		System.out.println("busy...");
> 		KThread.currentThread().yield();
> 	}
> 	
> 	child1.join();
> 	System.out.println("After joining, child1 should be finished.");
> 	System.out.println("is it? " + (child1.status == statusFinished));
> 	Lib.assertTrue((child1.status == statusFinished), "Expected child1 to be finished.");
> }
> ```

> [!info]- Sample test for `Condition2`
> ```java
> // Place Condition2 testing code in the Condition2 class
> // Example of the "interlock" pattern where two threads strictly alternate their execution with each other using a condition vaiable. (Also see the slide showing this pattern at the end of Lecture 6.)
> 
> private static class InterlockTest {
> 	private static Lock lock;
> 	private static Condition2 cv;
> 	
> 	private static class Interlocker implements Runnable {
> 		public void run() {
> 			lock.acquire();
> 			for(int i = 0; i < 10; i++){
> 				System.out.println(KThread.currentThread().getName());
> 				cv.wake();     // signal
> 				cv.sleep();    // wait
> 			}
> 			lock.release();
> 		}
> 	}
> 	
> 	public InterlockTest() {
> 		lcok = new Lock();
> 		cv = new Condition2(lock);
> 		
> 		KThread ping = new KThread(new Interlocker());
> 		ping.setName("ping");
> 		KThread pong = new KThread(new Interlocker());
> 		pong.setName("pong");
> 	
> 		ping.fork();
> 		pong.fork();
> 		
> 		// We need to wait for ping to finish, and the proper way to do so is to join on ping. (Note that when ping is done, pong is sleeping on the condition variable; if we were also to join pong, we would block forever.)
> 		
> 		// For this to work, join must be implemented. If you have not implemented join yet, then comment out the call to join and instead uncomment the loop with yields; the loop has the same effect, but is a kludgy way to do it.
> 		
> 		ping.join();
> 		// for(int i = 0; i < 50; i++){
> 			KThread.currentThread().yield();
> 		}
> 	}
> 	
> 	// Invoke Condition2.selfTest() from ThreadKernel.selfTest()
> 	public static void selfTest() {
> 		new InterlockTest();
> 	}
> }
> ```

> [!info]- More Complicated test for `Condition2`
> ```java
> // Place Condition2 test code inside of the Condition2 class.
>
> // Test programs should have exactly the same behavior with the
> // Condition and Condition2 classes. You can first try a test with
> // Condition, which is already provided for you, and then try it
> // with Condition2, which you are implementing, and compare their
> // behavior.
>
> // Do not use this test program as your first Condition2 test.
> // First test it with more basic test programs to verify specific
> // functionality.
>
> public static void cvTest5() {
>     final Lock lock = new Lock();
>     // final Condition empty = new Condition(lock);
>     final Condition2 empty = new Condition2(lock);
>     final LinkedList<Integer> list = new LinkedList<>();
>
>     KThread consumer = new KThread(new Runnable() {
>         public void run() {
>             lock.acquire();
>             while (list.isEmpty()) {
>                 empty.sleep();
>             }
>             Lib.assertTrue(list.size() == 5, "List should have 5 values.");
>             while (!list.isEmpty()) {
>                 KThread.currentThread().yield();
>                 System.out.println("Removed " + list.removeFirst());
>             }
>             lock.release();
>         }
>     });
>
>     KThread producer = new KThread(new Runnable() {
>         public void run() {
>             lock.acquire();
>             for (int i = 0; i < 5; i++) {
>                 list.add(i);
>                 System.out.println("Added " + i);
>                 KThread.currentThread().yield();
>             }
>             empty.wake();
>             lock.release();
>         }
>     });
>
>     consumer.setName("Consumer");
>     producer.setName("Producer");
>     consumer.fork();
>     producer.fork();
>
>     consumer.join();
>     producer.join();
>     // for (int i = 0; i < 50; i++) { KThread.currentThread().yield(); }
> }
> ```

> [!info]- Simple test for `sleepFor`
> ```java
> // Place sleepFor test code inside of the Condition2 class.
>
> private static void sleepForTest1() {
>     Lock lock = new Lock();
>     Condition2 cv = new Condition2(lock);
>
>     lock.acquire();
>     long t0 = Machine.timer().getTime();
>     System.out.println(KThread.currentThread().getName() + " sleeping");
>
>     // no other thread will wake us up, so we should time out
>     cv.sleepFor(2000);
>
>     long t1 = Machine.timer().getTime();
>     System.out.println(
>         KThread.currentThread().getName() +
>         " woke up, slept for " + (t1 - t0) + " ticks"
>     );
>
>     lock.release();
> }
>
> public static void selfTest() {
>     sleepForTest1();
> }
> ```

> [!info]- Simple test for `Rendezvous`
>  ```java
> // Place Rendezvous test code inside of the Rendezvous class.
>
> public static void rendezTest1() {
>     final Rendezvous r = new Rendezvous();
>
>     KThread t1 = new KThread(new Runnable() {
>         public void run() {
>             int tag = 0;
>             int send = -1;
>
>             System.out.println("Thread " + KThread.currentThread().getName() +
>                                " exchanging " + send);
>
>             int recv = r.exchange(tag, send);
>             Lib.assertTrue(recv == 1,
>                 "Was expecting " + 1 + " but received " + recv);
>
>             System.out.println("Thread " + KThread.currentThread().getName() +
>                                " received " + recv);
>         }
>     });
>     t1.setName("t1");
>
>     KThread t2 = new KThread(new Runnable() {
>         public void run() {
>             int tag = 0;
>             int send = 1;
>
>             System.out.println("Thread " + KThread.currentThread().getName() +
>                                " exchanging " + send);
>
>             int recv = r.exchange(tag, send);
>             Lib.assertTrue(recv == -1,
>                 "Was expecting " + -1 + " but received " + recv);
>
>             System.out.println("Thread " + KThread.currentThread().getName() +
>                                " received " + recv);
>         }
>     });
>     t2.setName("t2");
>
>     t1.fork();
>     t2.fork();
>
>     // assumes join is implemented correctly
>     t1.join();
>     t2.join();
> }
>
> // Invoke Rendezvous.selfTest() from ThreadedKernel.selfTest()
> public static void selfTest() {
>     // place calls to your Rendezvous tests that you implement here
>     rendezTest1();
> }
> ```

### Important: 
When implementing your own tests, if the main Nachos thread (the first thread that executes) ever returns or exits, then all of Nachos exits (even if other threads have not finished). This behavior is a peculiarity of how Nachos is implemented. If these semantics interfere with your tests then, if you have join implemented, have the main thread join on all children it creates (thereby guaranteeing that the main thread is the last one to finish). If you have only implemented Alarm, then in your Alarm tests have the main thread be the one that has the longest wait time (or you can have the main thread spin in test code until all children have woken up).

Nachos has a number of command line arguments, two of which are particularly useful for debugging this project:

- Invoking `nachos -d t` will display thread-related debugging information, such as context switches and thread state changes. You can add your own debugging output for this flag using `Lib.debug(dbgThread, ...)`.
- Invoking `nachos -s <number>` with different numbers will change when context switches happen.

Our grader will ignore output that you add with `Lib.debug` or `System.out.println`. You do not need to remove these for your submission (and some of them likely will come in handy for later projects), and you do not need to disable your tests.

During the project period, you can also use Gradescope to run a snapshot of your code on the six sample tests that we have provided above. Gradescope will then generate the output of running the tests in a file in your github repo. You can invoke Gradescope as many times as you like during the project period. Beyond these six sample tests, you are responsible for implementing your own tests to ensure that your code implements the remaining requirements for each task.