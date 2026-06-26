### 1. The Core Workflow
Git tracks changes through three main "zones":
1. **Working Directory:** Files you are currently editing.
2. **Staging Area (Index):** A "loading dock" for changes you want to include in your next snapshot.
3. **Repository (.git):** Where Git permanently stores the project history.

|Command|Action|
|---|---|
|`git init`|Initialize a new local Git repository.|
|`git status`|See which files are staged, unstaged, and untracked.|
|`git add <file>`|Move changes from the Working Directory to the Staging Area.|
|`git commit -m "msg"`|Save the staged snapshot to the version history.|

---
### 2. Branching & Merging
Branches allow you to work on new features without breaking the "main" code.
- **Create a branch:** `git branch <branch-name>`
- **Switch to a branch:** `git checkout <branch-name>` (or `git switch <branch-name>`)
- **Create & Switch:** `git checkout -b <branch-name>`
- **Merge into current:** `git merge <branch-name>`

---
### 3. Remote Repositories (GitHub/GitLab)
Used for syncing your local work with a server.
- **Connect to remote:** `git remote add origin <url>`
- **Upload changes:** `git push origin <branch-name>`
- **Download changes:** `git pull origin <branch-name>`
- **Clone a project:** `git clone <url>`

---

### 4. Undoing Mistakes

> [!CAUTION] Use these carefully, especially when working on shared branches.

- **Unstage a file:** `git reset <file>`
- **Discard local changes:** `git restore <file>`
- **Amend last commit:** `git commit --amend -m "new message"`
- **View history:** `git log --oneline --graph --all`

---
### Pro-Tip: The `.gitignore`

Always keep a `.gitignore` file in your root directory to prevent tracking junk files like `node_modules/`, `.DS_Store`, or compiled C binaries (e.g., `*.o`, `a.out`).