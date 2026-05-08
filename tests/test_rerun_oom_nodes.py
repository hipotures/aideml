from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node

from scripts.rerun_oom_nodes import (
    RERUN_PLAN_PREFIX,
    apply_execution_result_without_llm,
    select_oom_nodes,
)


def _oom_node(parent=None):
    node = Node(code="print('oom')", plan="plan", parent=parent)
    node._term_out = [
        "RuntimeError: REPL child process died unexpectedly\n",
        "CUDA error 2: out of memory\n",
    ]
    node.analysis = "CatBoost GPU ran out of memory"
    node.is_buggy = True
    node.status = "bug"
    return node


def test_select_oom_nodes_skips_attempted_sources():
    journal = Journal()
    first = _oom_node()
    second = _oom_node()
    journal.append(first)
    journal.append(second)
    records = [{"source_node_id": first.id, "rerun_step": 10}]

    selected = select_oom_nodes(journal, records=records)

    assert selected == [second]


def test_select_oom_nodes_ignores_recovery_rerun_nodes():
    journal = Journal()
    source = _oom_node()
    rerun = _oom_node(parent=source)
    rerun.plan = f"{RERUN_PLAN_PREFIX} of step 0"
    journal.append(source)
    journal.append(rerun)

    selected = select_oom_nodes(journal, records=[])

    assert selected == [source]


def test_apply_execution_result_without_llm_reads_autogluon_marker():
    node = Node(code="print('ok')", plan="plan")
    exec_result = ExecutionResult(
        term_out=[
            'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ok", '
            '"metric": 0.95108, "lower_is_better": false}\n'
        ],
        exec_time=2.0,
        exc_type=None,
    )

    apply_execution_result_without_llm(node, exec_result)

    assert node.is_buggy is False
    assert node.status == "ok"
    assert node.metric.value == 0.95108
    assert node.analysis == "ok"
