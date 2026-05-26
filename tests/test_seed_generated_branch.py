import json
from pathlib import Path

from aide.journal import Journal
from aide.journal import Node
from aide.utils import serialize
from scripts.seed_generated_branch import seed_generated_branch


def _write_hypothesis(
    root: Path,
    task: str,
    hypothesis_id: str,
    *,
    code: str | None = None,
    enabled: bool = True,
) -> None:
    hypothesis_dir = root / "research_hypotheses" / task / hypothesis_id
    hypothesis_dir.mkdir(parents=True)
    (hypothesis_dir / f"hypothesis-{hypothesis_id}.json").write_text(
        json.dumps(
            {
                "enabled": enabled,
                "agent_modes": ["legacy"],
                "title": f"Hypothesis {hypothesis_id}",
                "summary": "summary",
                "rationale": "rationale",
                "implementation_hint": "implementation",
                "expected_effect": "effect",
                "risk": "risk",
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    if code is not None:
        (hypothesis_dir / "legacy-001.py").write_text(code, encoding="utf-8")


def test_seed_generated_branch_preserves_root_then_child_order(tmp_path):
    task = "playground-series-s6e5"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / task
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task", encoding="utf-8")

    _write_hypothesis(repo_root, task, "001172", code="print('root old code')\n")
    _write_hypothesis(repo_root, task, "000806", enabled=False)
    _write_hypothesis(repo_root, task, "000530", enabled=False)

    def fake_child_generator(*, parent_node, selection, **_kwargs):
        hypothesis_id = selection.hypotheses[0].id
        node = Node(
            code=f"print('generated child {hypothesis_id}')\n",
            plan=f"generated patch {hypothesis_id}",
            parent=parent_node,
        )
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [hypothesis_id]
        node.research_source_hash = selection.source_hash
        return node

    result = seed_generated_branch(
        task=task,
        agent_mode="legacy",
        root_hypothesis="001172",
        root_code_file="legacy-001.py",
        children=("000806", "000530"),
        run_id="2-generated-branch-test",
        data_dir=data_dir,
        desc_file=desc_file,
        logs_dir=tmp_path / "logs",
        workspaces_dir=tmp_path / "workspaces",
        repo_root=repo_root,
        prepare_workspace=False,
        child_generator=fake_child_generator,
    )

    journal = serialize.load_json(result.log_dir / "journal.json", Journal)

    assert [node.research_hypotheses_offered[0] for node in journal.nodes] == [
        "001172",
        "000806",
        "000530",
    ]
    assert [node.status for node in journal.nodes] == [
        "generated",
        "generated",
        "generated",
    ]
    assert journal.nodes[0].parent is None
    assert journal.nodes[1].parent is journal.nodes[0]
    assert journal.nodes[2].parent is journal.nodes[0]
    assert journal.nodes[0].code == "print('root old code')\n"
    assert journal.nodes[1].code == "print('generated child 000806')\n"
    assert journal.nodes[2].code == "print('generated child 000530')\n"


def test_seed_generated_branch_does_not_require_child_code(tmp_path):
    task = "playground-series-s6e5"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / task
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task", encoding="utf-8")

    _write_hypothesis(repo_root, task, "001172", code="print('root old code')\n")
    _write_hypothesis(repo_root, task, "000806", enabled=False)

    def fake_child_generator(*, parent_node, selection, **_kwargs):
        node = Node(code="print('generated child')\n", plan="generated", parent=parent_node)
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [selection.hypotheses[0].id]
        node.research_source_hash = selection.source_hash
        return node

    result = seed_generated_branch(
        task=task,
        agent_mode="legacy",
        root_hypothesis="001172",
        root_code_file="legacy-001.py",
        children=("000806",),
        run_id="2-generated-branch-test",
        data_dir=data_dir,
        desc_file=desc_file,
        logs_dir=tmp_path / "logs",
        workspaces_dir=tmp_path / "workspaces",
        repo_root=repo_root,
        prepare_workspace=False,
        child_generator=fake_child_generator,
    )

    journal = serialize.load_json(result.log_dir / "journal.json", Journal)
    assert journal.nodes[1].code == "print('generated child')\n"
