# Hypothesis Mode Design

## Goal

Add a hard verification research mode for AIDE manual hypothesis registries.
The existing `llm` and `manual` modes remain unchanged:

- `research.mode=llm` keeps the current automatic LLM research behavior.
- `research.mode=manual` keeps the current soft manual-hints behavior.
- `research.mode=hypothesis` verifies exactly one selected hypothesis per
  generated node.

This mode is for empirical hypothesis testing, not for creating new hypotheses.

## Core Contract

In `hypothesis` mode, every generated node is assigned exactly one hypothesis ID.
The prompt must contain only that hypothesis and must instruct the coding agent
to implement that hypothesis, not choose from alternatives or invent an unrelated
experiment.

The review result must declare the same hypothesis ID. If the review result does
not contain the assigned ID, or contains a different ID, the node is a protocol
failure:

- set `status="failed"`;
- do not treat it as `bug`;
- do not allow descendants from that node.

`bug` remains reserved for implementation/runtime defects that can be debugged.
A debug child of a buggy node continues the same hypothesis.

## Selection Semantics

`research.every_steps=N` remains meaningful in `hypothesis` mode. It controls
periodic root exploration:

- every Nth step tries to create a new root with a new hypothesis;
- steps between root openings continue normal tree search and improve/debug
  existing nodes;
- if root hypotheses are exhausted, root creation is skipped and the run keeps
  expanding existing tree nodes.

Hypothesis uniqueness rules:

- root nodes use hypotheses that are globally unique among root nodes;
- an `ok -> improve` child must not reuse a hypothesis from its ancestor chain;
- an `ok -> improve` child must not duplicate a hypothesis already used by a
  direct sibling of the same parent;
- a `bug -> debug` child inherits the buggy node's hypothesis;
- a `failed` node is terminal and receives no children;
- the same hypothesis may be tested in different branches when it is not a root
  duplicate or sibling duplicate.

The selector should prefer under-tested compatible hypotheses, but must preserve
these uniqueness constraints.

## Metadata And Statistics

Each hypothesis-mode node records:

- `research_mode="hypothesis"`;
- exactly one offered hypothesis ID in `research_hypotheses_offered`;
- the same ID in `research_hypotheses_llm_claimed_used` only when the contract
  was satisfied;
- protocol failures as `status="failed"`.

The source hash is useful in logs/metadata, but it should not be shown inside
the prompt sent to the coding agent. The hypothesis ID is the visible contract.

This makes these statistics reliable for `hypothesis` runs:

- best/worst score per hypothesis;
- attempts per hypothesis;
- bug count per hypothesis;
- protocol failure count per hypothesis;
- global best score and the hypothesis that produced it.

Old `manual` runs can still expose offered/claimed usage, but they must not be
treated as hard hypothesis verification data.

## TUI Display

The run panel should expose the active or latest hypothesis ID in hypothesis
mode. Target formatting:

```text
Research   . 030 @ 000122 ✓
Best Score . 011 @ 19:13:00 0.95115 @ 000122
```

The tree should compactly attach the hypothesis ID to node outcomes:

```text
0.95104@000122
bug@000205
failed@000311
```

The compact tree format should be used only when a node has exactly one
hypothesis ID. Existing display behavior remains for other modes.

## Testing Scope

Tests should cover:

- `research.mode=hypothesis` selects exactly one hypothesis;
- root selection obeys `research.every_steps`;
- root hypotheses are globally unique;
- improve children avoid ancestor and sibling hypothesis duplicates;
- debug children inherit the same hypothesis;
- missing or wrong claimed hypothesis ID marks the node as `failed`, not `bug`;
- failed nodes remain terminal;
- TUI formatting appends `@hypothesis_id` for exactly-one-hypothesis nodes;
- `llm` and `manual` behavior remains unchanged.
