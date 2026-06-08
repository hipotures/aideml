"""
The journal is the core datastructure in AIDE that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from dataclasses_json import DataClassJsonMixin
from .interpreter import ExecutionResult
from .utils.metric import MetricValue
from .utils.response import trim_long_string


OOM_FAILURE_PARENT_LIMIT = 3
SEEDED_BASE_SUMMARY_PREFIX = "Seeded base artifact"
PUBLIC_CV_CLOSE_TOLERANCE_PCT = 0.05
PUBLIC_CV_MEANINGFUL_TOLERANCE_PCT = 0.20


def _summary_analysis_text(analysis: str | None) -> str:
    text = str(analysis or "")
    if text.startswith("AutoGluon preprocess wrapper completed"):
        return ""
    return re.sub(r"\s+using presets=[^.]+", "", text)


def _summary_plan_text(plan: str | None) -> str:
    text = str(plan or "")
    if text.startswith("External Codex synthesis checkpoint"):
        return "External Codex synthesis generated a new root solution from top candidates."
    if text.startswith(SEEDED_BASE_SUMMARY_PREFIX):
        marker = " Original plan: "
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return "Seeded base artifact copied from a previous scored run."
    return text


def _format_public_cv_interpretation(
    *,
    cv_score: float,
    public_score: float,
    maximize: bool,
) -> str:
    raw_gap = public_score - cv_score
    if cv_score == 0:
        relative_pct = None
    else:
        oriented_gap = raw_gap if maximize else -raw_gap
        relative_pct = 100.0 * oriented_gap / abs(cv_score)

    lines = [
        f"Kaggle public score: {public_score:.5f}",
        f"CV-public gap: {raw_gap:+.5f}.",
    ]
    reliability_prefix = "Use public score only as reliability context, not as an optimization target. "

    if relative_pct is None:
        lines.append(
            "Public/CV interpretation: CV is zero, so percentage alignment is undefined. "
            + reliability_prefix
            + "Inspect the raw public/CV gap before making the next change."
        )
        return "\n".join(lines)

    abs_pct = abs(relative_pct)
    if abs_pct <= PUBLIC_CV_CLOSE_TOLERANCE_PCT:
        if raw_gap == 0:
            comparison = "the same as CV"
        else:
            direction = "lower" if raw_gap < 0 else "higher"
            comparison = f"{abs_pct:.2f}% {direction} than CV"
        lines.append(
            f"Public/CV interpretation: public is {comparison}; public and CV look close/aligned. "
            + reliability_prefix
            + "Continue with the intended hypothesis normally while preserving CV discipline."
        )
    elif relative_pct < 0:
        severity = (
            "mildly lower"
            if abs_pct <= PUBLIC_CV_MEANINGFUL_TOLERANCE_PCT
            else "meaningfully lower"
        )
        action = (
            "Prefer a conservative robustness-oriented change and avoid adding major "
            "model complexity unless the hypothesis directly addresses generalization."
            if abs_pct <= PUBLIC_CV_MEANINGFUL_TOLERANCE_PCT
            else "Prefer diagnostic, validation, leakage-checking, simplification, "
            "or robustness changes; do not stack more complexity on this node just because CV is high."
        )
        if maximize:
            comparison = "lower than CV"
        else:
            comparison = "worse than CV"
        lines.append(
            f"Public/CV interpretation: public is {abs_pct:.2f}% {comparison}; "
            f"{severity}, possible overfitting or CV/public split mismatch. "
            + reliability_prefix
            + action
        )
    else:
        if maximize:
            comparison = "higher than CV"
        else:
            comparison = "better than CV"
        lines.append(
            f"Public/CV interpretation: public is {abs_pct:.2f}% {comparison}; "
            "CV may be conservative or the public split favorable. "
            + reliability_prefix
            + "Do not chase the public leaderboard directly; preserve CV discipline and generalization."
        )

    return "\n".join(lines)


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    code: str
    code_path: str | None = field(default=None, kw_only=True)
    plan: str = field(default=None, kw_only=True)  # type: ignore

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    artifact_dir_name: str | None = field(default=None, kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)
    status: Literal["ok", "bug", "failed", "generated"] | None = field(
        default=None,
        kw_only=True,
    )

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)
    run_stats: dict | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    validity_warning: str | None = field(default=None, kw_only=True)
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore
    submission_validation: dict | None = field(default=None, kw_only=True)
    research_mode: str | None = field(default=None, kw_only=True)
    research_hypotheses_offered: list[str] = field(default_factory=list, kw_only=True)
    research_source_hash: str | None = field(default=None, kw_only=True)
    research_hypotheses_llm_claimed_used: list[str] = field(
        default_factory=list,
        kw_only=True,
    )
    research_usage_note: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    @property
    def is_submission_contract_error(self) -> bool:
        if self.status == "generated":
            return False
        validation = self.submission_validation or {}
        return (
            self.exc_type == "SubmissionValidationError"
            or validation.get("status") == "error"
        )

    @property
    def is_oom_failure(self) -> bool:
        text = f"{self.analysis or ''}\n{''.join(self._term_out or [])}"
        return (
            "CatBoost GPU ran out of memory" in text
            or "CUDA error 2: out of memory" in text
        )

    @property
    def is_gpu_execution_failure(self) -> bool:
        text = f"{self.analysis or ''}\n{''.join(self._term_out or [])}"
        return self.is_oom_failure or (
            "LightGBM CUDA native crash" in text
            or "NVRM: Xid" in text
            or "MMU Fault" in text
            or "FAULT_PDE" in text
        )

    @property
    def is_terminal_failure(self) -> bool:
        return self.status == "failed" or self.is_gpu_execution_failure or (
            self.status is None
            and self.is_buggy
            and str(self.code or "").lstrip().startswith("# Failed ")
        )

    @property
    def oom_failure_child_count(self) -> int:
        return sum(1 for child in self.children if child.is_oom_failure)

    @property
    def is_oom_blocked_parent(self) -> bool:
        return self.oom_failure_child_count >= OOM_FAILURE_PARENT_LIMIT

    @property
    def is_in_submission_contract_error_branch(self) -> bool:
        node: Node | None = self
        while node is not None:
            if node.is_submission_contract_error:
                return True
            node = node.parent
        return False

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore


@dataclass
class InteractiveSession(DataClassJsonMixin):
    """
    A collection of nodes for an interaction session
    (when the agent interacts with a Jupyter notebook-like interface).
    """

    nodes: list[Node] = field(default_factory=list)
    completed: bool = False

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    def generate_nb_trace(self, include_prompt, comment_headers=True) -> str:
        """Generate a trace of the interactive session in IPython format."""
        trace = []
        header_prefix = "## " if comment_headers else ""
        for n in self.nodes:
            trace.append(f"\n{header_prefix}In [{n.step + 1}]:\n")
            trace.append(n.code)
            trace.append(f"\n{header_prefix}Out [{n.step + 1}]:\n")
            trace.append(n.term_out)

        if include_prompt and self.nodes:
            trace.append(f"\n{header_prefix}In [{self.nodes[-1].step + 2}]:\n")

        return "\n".join(trace).strip()


@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)
    # eda: InteractiveSession = field(default_factory=lambda: InteractiveSession())

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if not n.is_buggy and n.status != "generated"]

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good=True) -> None | Node:
        """Return the best solution found so far (node with the highest validation metric)."""
        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes
        nodes_with_metrics = [
            n for n in nodes if n.metric is not None and n.metric.value is not None
        ]
        if not nodes_with_metrics:
            return None
        return max(nodes_with_metrics, key=lambda n: n.metric)

    def generate_summary(
        self,
        include_code: bool = False,
        recent_steps: int | None = None,
        full_recent_steps: int | None = None,
        public_scores_by_node_id: dict[str, float] | None = None,
    ) -> str:
        """Generate a summary of the journal for the agent."""
        public_scores_by_node_id = public_scores_by_node_id or {}
        good_nodes = self.good_nodes
        best_node = self.get_best_node()
        steps = [
            n.step
            for n in good_nodes
            if isinstance(n.step, int) and not isinstance(n.step, bool)
        ]

        included_steps = None
        if recent_steps is not None:
            if recent_steps <= 0:
                included_steps = set()
            elif steps:
                included_steps = set(sorted(set(steps))[-recent_steps:])

        full_steps = None
        if full_recent_steps is not None:
            if full_recent_steps <= 0:
                full_steps = set()
            elif steps:
                full_steps = set(sorted(set(steps))[-full_recent_steps:])

        summary = []
        for n in good_nodes:
            has_step = isinstance(n.step, int) and not isinstance(n.step, bool)
            if included_steps is not None and (not has_step or n.step not in included_steps):
                continue
            include_full_node = full_steps is None or (
                isinstance(n.step, int)
                and not isinstance(n.step, bool)
                and n.step in full_steps
            )
            summary_part = f"Design: {_summary_plan_text(n.plan)}\n"
            if include_code and include_full_node:
                summary_part += f"Code: {n.code}\n"
            if include_full_node:
                analysis_text = _summary_analysis_text(n.analysis)
                if analysis_text:
                    summary_part += f"Results: {analysis_text}\n"
                if n.validity_warning:
                    summary_part += f"Validity warning: {n.validity_warning}\n"
            best_marker = " (current best)" if n is best_node else ""
            summary_part += f"Validation Metric: {n.metric.value:.5f}{best_marker}\n"
            public_score = public_scores_by_node_id.get(n.id)
            if public_score is not None:
                summary_part += (
                    _format_public_cv_interpretation(
                        cv_score=float(n.metric.value),
                        public_score=public_score,
                        maximize=n.metric.maximize is not False,
                    )
                    + "\n"
                )
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)

    def _ancestor_path(self, node: Node) -> list[Node]:
        path: list[Node] = []
        current: Node | None = node
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def generate_branch_context(
        self,
        parent_node: Node,
        *,
        public_scores_by_node_id: dict[str, float] | None = None,
    ) -> str:
        """Generate hypothesis-mode context for the selected parent branch."""
        public_scores_by_node_id = public_scores_by_node_id or {}
        ancestors = self._ancestor_path(parent_node)
        hypothesis_path = [
            n.research_hypotheses_offered[0]
            for n in ancestors
            if len(n.research_hypotheses_offered) == 1
        ]
        lines = [
            (
                "The previous code is the current parent code. The entries below "
                "are the ancestor nodes of this parent, ordered from root to direct parent."
            ),
            "They describe earlier hypotheses already incorporated into the previous code.",
            "",
        ]
        if hypothesis_path:
            lines.extend(["Branch path:", " -> ".join(hypothesis_path), ""])

        for idx, ancestor in enumerate(ancestors, start=1):
            labels = []
            if idx == 1:
                labels.append("root")
            if idx == len(ancestors):
                labels.append("direct parent")
            label_suffix = f" / {' / '.join(labels)}" if labels else ""
            lines.append(f"Ancestor {idx}{label_suffix}:")
            if len(ancestor.research_hypotheses_offered) == 1:
                lines.append(
                    f"Hypothesis ID: {ancestor.research_hypotheses_offered[0]}"
                )
            lines.append(f"Design: {_summary_plan_text(ancestor.plan)}")
            if ancestor.metric is not None and ancestor.metric.value is not None:
                lines.append(f"Validation Metric: {ancestor.metric.value:.5f}")
                public_score = public_scores_by_node_id.get(ancestor.id)
                if public_score is not None:
                    lines.extend(
                        _format_public_cv_interpretation(
                            cv_score=float(ancestor.metric.value),
                            public_score=public_score,
                            maximize=ancestor.metric.maximize is not False,
                        ).splitlines()
                    )
            if idx != len(ancestors):
                lines.extend(["", "-------------------------------"])
            lines.append("")

        lines.extend(
            [
                "Instruction: Use the previous code as the authoritative current state.",
                "This branch context only describes earlier changes already present in that code.",
                "Preserve these earlier branch changes unless they directly conflict with the assigned hypothesis.",
            ]
        )
        return "\n".join(lines).strip()
