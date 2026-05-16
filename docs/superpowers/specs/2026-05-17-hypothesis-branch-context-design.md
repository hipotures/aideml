# Hypothesis Branch Context Design

## Goal

In `research.mode=hypothesis`, code generation must verify one assigned
hypothesis against the current parent code. The prompt should stop showing the
global journal `Memory`, because that mixes independent histories and can make
the model reuse unrelated ideas instead of implementing the assigned hypothesis.

Replace global `Memory` only in `hypothesis` mode with branch-local context.
Existing `llm` and `manual` research modes keep their current global memory
behavior.

## Scope

This change affects prompt construction only:

- Draft/root prompts in `hypothesis` mode omit `Memory` entirely.
- Child/improvement/debug prompts in `hypothesis` mode include `Branch context`
  built from the ancestor path of the selected parent node.
- `Previous solution` and `Previous preprocess function` remain the
  authoritative current code state.
- `Hypothesis under verification` remains unchanged.
- Hypothesis selection, review contract, usage statistics, TUI, and execution
  behavior remain unchanged.

## Branch Context Format

For a root node, omit the section entirely. Do not include filler text such as
`No previous branch context`; the model does not need tree metadata when there
is no parent.

For a child node, include a section like:

```markdown
# Branch context

The previous code is the current parent code. The entries below are the ancestor
nodes of this parent, ordered from root to direct parent. They describe earlier
hypotheses already incorporated into the previous code.

Branch path:
000418 -> 000880 -> 000731

Ancestor 1 / root:
Hypothesis ID: 000418
Design: For hypothesis `000418`, ...
Validation Metric: 0.94907

-------------------------------
Ancestor 2:
Hypothesis ID: 000880
Design: Using hypothesis `000880`, ...
Validation Metric: 0.94911

-------------------------------
Ancestor 3 / direct parent:
Hypothesis ID: 000731
Design: For hypothesis `000731`, ...
Validation Metric: 0.94922

Instruction: Use the previous code as the authoritative current state. This
branch context only describes earlier changes already present in that code.
Preserve these earlier branch changes unless they directly conflict with the
assigned hypothesis.
```

The hierarchy must be explicit. A plain list of designs is not enough, because
the model should know that these are ancestors ordered from root to direct
parent, not independent candidate ideas.

## Content Rules

Each ancestor entry should include:

- ancestor position label, including `root` for the first and `direct parent`
  for the last;
- `Hypothesis ID` when the node has exactly one offered hypothesis;
- compact design/plan text from the node;
- validation metric when available.

Do not include ancestor code. The parent code is already included separately as
the previous code. Repeating ancestor code would enlarge prompts and create
conflicting sources of truth.

Do not include siblings, descendants, unrelated roots, or all good nodes from
the journal. The context must be the selected parent branch only.

## Data Flow

Prompt construction should use a helper that walks from `parent_node` to root,
reverses the path, and renders the entries in root-to-parent order.

If `research.mode != "hypothesis"`, keep the existing `self.journal.generate_summary()`
usage.

If `research.mode == "hypothesis"`:

- root draft: no `Memory`, no `Branch context`;
- child improve/debug: `Branch context` from the parent ancestor path.

## Tests

Add tests that verify:

- hypothesis root draft prompts omit `Memory` and omit `Branch context`;
- hypothesis child prompts include `Branch context` ordered root to direct
  parent;
- branch context includes only ancestors of the selected parent, not siblings or
  unrelated roots;
- legacy/manual or LLM research prompt behavior still includes the existing
  global `Memory` where it did before.

