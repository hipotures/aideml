## IMPORTANT: Avoid commands that cause output buffering issues
- DO NOT pipe output through `head`, `tail`, `less`, or `more` when monitoring or checking command output
- DO NOT use `| head -n X` or `| tail -n X` to truncate output - these cause buffering problems
- Instead, let commands complete fully, or use `--max-lines` flags if the command supports them
- For log monitoring, prefer reading files directly rather than piping through filters

## Python and uv
- Always run project Python commands through `uv run python`, including one-off probes, `-c` snippets, module commands, and syntax checks.
- Use `uv run python -m py_compile ...` instead of plain `python -m py_compile ...`.
- Use `uv run pytest`, `uv run ruff`, and other `uv run ...` commands for project tools.
- Do not use plain `python` unless explicitly checking the system interpreter outside the project environment.
- uv pip, never pip!

## When checking command output:
- Run commands directly without pipes when possible
- If you need to limit output, use command-specific flags (e.g., `git log -n 10` instead of `git log | head -10`)
- Avoid chained pipes that can cause output to buffer indefinitely

## Git workflow
- If you modify files in a Git repository, do not finish the task with uncommitted changes unless the user explicitly says not to commit.
- Any task that changes files must end in one of two states: changes committed, or an explicit explanation why they were not committed.
- Before committing, run relevant verification and inspect `git status --short`.
- Commit only changes made for the current task.
- Never commit unrelated user changes.
- Use concise commit messages.
