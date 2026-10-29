## Common Mistakes to Avoid
```c
// Lessons:
//   1  Use-after-free: handing a pointer to code that frees it
//   2  Memory leaks: hidden allocations, "= NULL" is not free
//   3  Uninitialized memory: strcat and malloc
//   4  Signed vs. unsigned: a check that can never be true
//   5  sizeof the wrong thing: char vs. char*
//   6  Reading a helper's contract: argument order

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ======================================================================
// Lesson 1: Use-after-free
// ======================================================================

static void print_and_free(int* n) {
    printf("printing %d\n", *n);
    free(n);
}

static void lesson1_buggy(void) {
    int* score = malloc(sizeof *score);
    *score = 42;

    print_and_free(score);
    printf("score is still %d\n", *score);   // BUG
}

static void lesson1_fixed(void) {
    int* score = malloc(sizeof *score);
    *score = 42;

    int saved = *score;      // copy what you need BEFORE giving it away
    print_and_free(score);
    score = NULL;            // now any accidental use crashes loudly
    printf("score was %d\n", saved);
}

// ======================================================================
// Lesson 2: Memory leaks
// ======================================================================

static char* copy_string(const char* s) {
    char* copy = malloc(strlen(s) + 1);
    strcpy(copy, s);
    return copy;
}

static void lesson2_buggy(void) {
    // BUG
    printf("length: %zu\n", strlen(copy_string("hello")));

    // BUG
    int* nums = malloc(10 * sizeof *nums);
    nums = NULL;
}

static void lesson2_fixed(void) {
    char* copy = copy_string("hello");
    printf("length: %zu\n", strlen(copy));
    free(copy);

    int* nums = malloc(10 * sizeof *nums);
    free(nums);     // free first...
    nums = NULL;    // ...then clear
}

// ======================================================================
// Lesson 3: Uninitialized memory
// ======================================================================

static void lesson3_buggy(void) {
    char name[16];                  // BUG
    strcat(name, "Bob");
    printf("name: %s\n", name);

    int* count = malloc(sizeof *count);   // BUG
    *count += 1;
    printf("count: %d\n", *count);
    free(count);
}

static void lesson3_fixed(void) {
    char name[16] = "";             // starts as an empty string
    strcat(name, "Bob");
    printf("name: %s\n", name);

    int* count = calloc(1, sizeof *count);   // calloc zeroes the memory
    *count += 1;
    printf("count: %d\n", *count);
    free(count);
}

// ======================================================================
// Lesson 4: Signed vs. unsigned
// ======================================================================

static void lesson4_buggy(void) {
    uint8_t change = 5 - 8;         // BUG

    if (change < 0) {
        printf("went down\n");
    } else {
        printf("went up by %d\n", change);
    }
}

static void lesson4_fixed(void) {
    int8_t change = 5 - 8;

    if (change < 0) {
        printf("went down\n");
    } else {
        printf("went up by %d\n", change);
    }
}

// ======================================================================
// Lesson 5: sizeof the wrong thing
// ======================================================================

static void lesson5_buggy(void) {
    // BUG
    char** names = malloc(3 * sizeof(char));

    names[0] = "Ann";
    names[1] = "Bo";
    names[2] = "Cy";
    printf("%s %s %s\n", names[0], names[1], names[2]);
    free(names);
}

static void lesson5_fixed(void) {
    char** names = malloc(3 * sizeof(char*));

    names[0] = "Ann";
    names[1] = "Bo";
    names[2] = "Cy";
    printf("%s %s %s\n", names[0], names[1], names[2]);
    free(names);
}

// ======================================================================
// Lesson 6: Read the helper's contract
// ======================================================================

// Returns deadline - now. Positive means there is still time left.
static int time_left(int now, int deadline) {
    return deadline - now;
}

static void lesson6_buggy(void) {
    int now = 10, deadline = 20;

    if (time_left(now, deadline) >= 0) {   // BUG
        printf("expired\n");
    } else {
        printf("still time\n");
    }
}

static void lesson6_fixed(void) {
    int now = 10, deadline = 20;

    if (time_left(now, deadline) <= 0) {
        printf("expired\n");
    } else {
        printf("still time\n");
    }
}
```

## Directions
![[CSE123 Fall 2026 PA1a Specification.pdf]]