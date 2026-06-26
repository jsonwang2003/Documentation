# Minimum Viable Product
The simplest functional version of a product that delivers core value to early adopters while enabling teams to validate assumptions and gather feedback with minimal resources.

# Project: Agent Issue Tracker

## MUST HAVE (without not an AIT that meets requirements)
 - AI agent reads User Issues
	 - User creates the issues (in our app, github issues, or something else)
 - AI reasons through the Issues
	 - Output as a summary of what it thinks seeks approval from user before jumping to action
 - Displays Issues to User
	 - Time Created / Updated / Resolved
	 - Title
	 - Description
 - User controls token count / cost
	 - Needs easy UI/UX for User to provide issue changes

### CRUD(Create, Read, Update, Delete)
- AI doesn't touch any of these functions of the software, these should be handled by human supervisor
- This ensures the issues and task flow not being disrupted by AI mistakes
	- This does not ensure the user trusting the AI, but rather a safe guard for AI not disrupting anything pre-existing

### Workflow Diagram

## SHOULD HAVE (makes it a good product)
-  AI-assigned priority fields (e.g. p1, p2, p3) OR size (XS, S, M, L, XL) on issues
- Assignment of which agent contributors / human supervisors to the issue (helps debugging)
## NICE TO HAVE (we will ignore these) 
- some kind of graph visualization (one or some of: dependencies, issue status proportions, etc)
- Features which help the user trust the AI
- AI context read on prior issues and codebase
- Skills document setting on AI Agent for user to customize
- Reasoning trace snapshot
- Prompt version and model version metadata
- Making the problem-solving process repeatable
- Duplicate detection, Similarity Search, Linked Historical Issues
- Making the process of managing many agents easier (less of a blackbox)
- Filtering by failure codes, agent behavioral patterns
- Loop detection / Agent timeout
- Token / Cost guardrails