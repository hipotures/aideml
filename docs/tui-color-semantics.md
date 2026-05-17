# TUI Color Semantics

The AIDE terminal UI uses color as a data category, not as decoration. The goal is
to keep long-running experiment screens scannable while avoiding a mixed palette
where labels, metrics, paths, statuses, and errors all compete for attention.

## Categories

| Category | Examples | Style |
| --- | --- | --- |
| Panel title | `Run data`, `Logs` | bold white, handled by the panel title |
| Static section label | `Models`, `Base path`, `Resources` | bold cyan |
| Static row label | `Research`, `Phase`, `Best Score`, model names | bold cyan |
| Separator | `·`, spacing punctuation | dim |
| Neutral dynamic value | paths, model ids, reasoning effort | yellow |
| Operator notice | graceful `Ctrl+C` wait message | yellow |
| Metric dynamic value | best score step, timestamp, score, hypothesis id | green |
| Running dynamic status | active checkpoint step, running symbol | cyan |
| Failed dynamic status | failed checkpoint step, failure symbol | red |
| Active hypothesis phase | the currently controlling phase counter | green |
| Inactive hypothesis phase | the other phase counter | dim |
| Error heading | `Last Error` | bold red |
| Error body | traceback or exception text | dim |
| Log hint labels | `Hypothesis`, `Title`, `Summary`, `Try` | bold cyan |
| Log hint values | hypothesis text | dim |

## Run Data Lines

Run-data status lines should keep their row label stable and color only the
dynamic value segment:

```text
◇ Research   · 149 @ 000847 ✓
◇ Phase      · exploration 33/50 · exploitation 116/1450
★ Best Score · 145 @ 19:59:14 0.95193 · 000011
```

`Research`, `Phase`, and `Best Score` are labels, so they use the static row
label color. The separator is dim. The status, phase counter, or metric segment
uses the semantic dynamic color.

## Hypothesis Phase Counters

`agent.steps` is the total run budget:

```text
agent.steps = exploration budget + exploitation budget
```

In `research.mode=hypothesis`, exploration means root-hypothesis verification.
Exploitation means branch work after the root limit has been reached. Because a
resume can increase the root limit after exploitation has already started, the
TUI always shows both counters. The active phase is green and the inactive phase
is dim.

If a resumed run lowers `research.hypothesis_root_limit` below the number of
already tested roots, the TUI uses the already tested root count as the effective
exploration budget. That avoids misleading displays like `150/100`.

## Operator Notices

Operator notices are user-action messages that need attention but are not
failures. The main example is the first `Ctrl+C` during code execution:

```text
Ctrl+C received. Waiting for current code to finish. The node will be reviewed
and saved, then the run will stop. Press Ctrl+C again to stop now.
```

Use yellow for the full notice. Yellow is reserved here for warning-like
operational state and neutral values such as paths or model identifiers; it
should not be used for metric labels or success states.

## What Not To Do

- Do not color an entire row green just because its status is successful.
- Do not use yellow for labels and values in the same row.
- Do not treat operator notices as errors; reserve red for actual failure
  states and error headings.
- Do not make `Phase` a special visual style; it is a normal static row label.
- Do not recolor error bodies red; the red heading is enough and the body should
  stay readable.
