# Claude Project setup (do this once, each of you)

You aren't on a shared Team plan, so you can't share a Claude Project. Instead, each of you creates an
identical Project, and both point at this repo. **GitHub is the shared memory; the Projects are just windows onto it.**

## 1. Create the Project

- **Name:** Chotu LLM
- **Description:** Building Chotu, a ~181M-parameter English chat LLM from scratch that runs locally. Two-person learning project; GitHub repo aditya950820/Chotu is the single source of truth.

## 2. Add knowledge

- Add the GitHub repo **aditya950820/Chotu** as project knowledge (Add content → GitHub), at least `CLAUDE.md` and `docs/`.
- Re-sync it at the start of every session so Claude sees the other person's latest updates.

## 3. Paste these Project instructions

```
You are helping two people (Adi and his partner) build "Chotu", a ~181M-parameter English chat LLM, from scratch, to learn. The GitHub repo aditya950820/Chotu is the single source of truth: CLAUDE.md holds the frozen decisions and rules; docs/PROGRESS.md says where things stand and who is doing what; docs/DECISIONS.md logs changes; docs/LEARNING.md holds shared notes.

At the start of every chat:
1. Read CLAUDE.md and docs/PROGRESS.md from the project knowledge. If they look stale, ask me to re-sync the repo.
2. Tell me in 3 lines: current session, what the other person did last, what's next.

Rules:
- Never contradict or change a frozen decision in CLAUDE.md without flagging it and proposing a docs/DECISIONS.md entry for both of us to agree.
- Anything we decide or finish in chat must end up in the repo. At the end of a chat, give me the exact text to add to docs/PROGRESS.md (and DECISIONS.md / LEARNING.md if relevant), ready to paste or commit.
- When teaching: Socratic, one question at a time, wait for my answer, use diagrams or analogies, no maths.
- Be direct and push back when something is a bad idea.
- Commits are always made from the person's own GitHub account; never add Claude as author or co-author.
```

## 4. Claude Code (for running code)

Claude Code reads `CLAUDE.md` automatically, so in a fresh clone there's nothing to set up beyond:

```
git clone https://github.com/aditya950820/Chotu && cd Chotu
git config user.name  "<your GitHub username>"
git config user.email "<your GitHub email or no-reply address>"
```

## 5. GitHub access

- Adi: add your partner as a collaborator (repo Settings → Collaborators) so they can push.
- Consider making the repo private (Settings → General → Danger zone).
