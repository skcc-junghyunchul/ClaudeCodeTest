# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

GitHub: https://github.com/skcc-junghyunchul/ClaudeCodeTest
Git user: skcc-junghyunchul / charliejung87@gmail.com

## Git workflow

Commit and push to `origin/main` continuously throughout work — not just at the end. The goal is that no progress is ever lost and the GitHub history always reflects the current state of the project.

- Commit after every logical unit of work: a new feature, a bug fix, a styling change, a new file
- Never batch multiple unrelated changes into a single commit
- Write commit messages in the imperative mood with a short subject line, followed by a blank line and bullet points if the change needs more explanation
- Always `git push` immediately after committing — do not leave commits sitting unpushed

## Running the project

This is a no-build, no-dependency project. Open any HTML file directly in a browser:

```bash
open tictactoe.html
```

## Architecture

Each game or experiment is a **single self-contained HTML file** — HTML structure, CSS, and JavaScript all in one file. There is no build step, no package manager, and no framework.

### tictactoe.html

- Board state: flat `Array(9)` indexed 0–8 (row-major order)
- Win detection: checked against the 8 hardcoded `WINS` combinations after every move
- Scores (`{ X, O, draw }`) live in memory and reset on page reload — they are not persisted
- CSS classes drive visual state: `.taken` blocks re-clicks, `.win` highlights the winning line, `.x`/`.o` set player colors

## Conventions

- Keep each game or tool as a single `.html` file — no external dependencies
- Color palette: background `#1a1a2e`, X color `#e94560`, O color `#a8dadc`
