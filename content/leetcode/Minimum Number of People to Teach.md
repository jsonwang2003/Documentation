---
tags:
  - Arrays
  - HashTable
  - Greedy
---
## Description
[Minimum Number of People to Teach - LeetCode](https://leetcode.com/problems/minimum-number-of-people-to-teach/description/?envType=daily-question&envId=2025-09-10)

On a social network consisting of `m` users and some friendships between users, two users can communicate with each other if they know a common language.

You are given an integer `n`, an array `languages`, and an array `friendships` where:

- There are `n` languages numbered `1` through `n`,
- `languages[i]` is the set of languages the `i​​​​​​th`​​​​ user knows, and
- `friendships[i] = [u​​​​​​i​​​, v​​​​​​i]` denotes a friendship between the users `u​​​​​​​​​​​i`​​​​​ and `vi`.

You can choose **one** language and teach it to some users so that all friends can communicate with each other. Return _the_ _**minimum**_ _number of users you need to teach._

Note that friendships are not transitive, meaning if `x` is a friend of `y` and `y` is a friend of `z`, this doesn't guarantee that `x` is a friend of `z`.

## Examples
#### **Example 1:**

**Input:** n = 2, languages = `[[1],[2],[1,2]]`, friendships = `[[1,2],[1,3],[2,3]]`
**Output:** 1
**Explanation:** You can either teach user 1 the second language or user 2 the first language.

#### **Example 2:**

**Input:** n = 3, languages = `[[2],[1,3],[1,2],[3]]`, friendships = `[[1,4],[1,2],[3,4],[2,3]]`
**Output:** 2
**Explanation:** Teach the third language to users 1 and 3, yielding two users to teach.

## Constraints
- `2 <= n <= 500`
- `languages.length == m`
- `1 <= m <= 500`
- `1 <= languages[i].length <= n`
- `1 <= languages[i][j] <= n`
- `1 <=` $u_i$ `<` $v_i$ `<= languages.length`
- `1 <= friendships.length <= 500`
- All tuples $(u_i, v_i)$ are unique
- `languages[i]` contains only unique values

## Code
```cpp
class Solution {
public:
    int minimumTeachings(int n, vector<vector<int>>& languages, vector<vector<int>>& friendships) {
        // Store all the friends who cannot communicate
        unordered_set<int> noComm;
        for (auto friendship : friendships){
            unordered_map<int, int> map;
            bool canComm = false;

            for (auto lan : languages[friendship[0] - 1]){
                // load in the langauges user1 speaks
                map[lan] = 1;
            }
            for (auto lan: languages[friendship[1] - 1]){
                if (map[lan]) {
                    // The user2 speaks a common language with user1
                    canComm = true;
                    break;
                }
            }

            if (!canComm) {
                noComm.insert(friendship[0] - 1);
                noComm.insert(friendship[1] - 1);
            }
        }

        int maxCount = 0;
        vector<int> count(n + 1, 0);
        for (auto user : noComm){
            for(int lan : languages[user]){
                count[lan]++;
                maxCount = max(maxCount, count[lan]);
            }
        }

        return noComm.size() - maxCount;
    }
};
```

## Approach
1. Find out all the users who has a friendship that cannot communicate with
	- Store all of them in a set `noComm`
	- Loop through each friendship link
		- Store all the languages the first user in the friendship can speak in a hashmap `map`
		- Determine if the second user in the friendship can speak a language that the first user speaks
			- True: break and update a Boolean variable `canComm` to store it
			- False: continue until no common language found
	- If the 2 users in the friendship does not have a common tongue, add them into the set
2. Find out the language that has the most amount of speakers
	- Loop through each user that has at least 1 link that does not have a common tongue with their friend
		- Loop through each language that the user speaks
		- Add them into a count
		- Update if there is a new language that has the most speaker
	- Resulting the count of a language that most user speaks
3. Return the number of users - the amount of users that has the most speakers