import re

from aide.journal import Journal
from aide.run import load_resume_state, next_generated_only_node
from aide.utils import serialize
from scripts.seed_manual_legacy import seed_manual_legacy_run


def test_seed_manual_legacy_creates_resumable_generated_drafts(tmp_path, monkeypatch):
    monkeypatch.setenv("AIDE_AGENT_STEPS", "1000")
    monkeypatch.setenv("AIDE_AGENT_MODE", "autogluon_preprocess")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("manual legacy task\n", encoding="utf-8")
    first = tmp_path / "01_first.py"
    second = tmp_path / "02_second.py"
    first.write_text("print('first')\n", encoding="utf-8")
    second.write_text("print('second')\n", encoding="utf-8")

    result = seed_manual_legacy_run(
        sources=(first, second),
        data_dir=data_dir,
        desc_file=desc_file,
        logs_dir=tmp_path / "logs",
        workspaces_dir=tmp_path / "workspaces",
        prepare_workspace=False,
    )

    journal = serialize.load_json(result.log_dir / "journal.json", Journal)
    assert [node.status for node in journal.nodes] == ["generated", "generated"]
    assert [node.step for node in journal.nodes] == [0, 1]
    assert [node.code for node in journal.nodes] == [
        "print('first')\n",
        "print('second')\n",
    ]
    for node in journal.nodes:
        assert re.fullmatch(
            r"\d{8}T\d{6}-[0-9a-f]{8}-\d+", node.artifact_dir_name or ""
        )
        assert (
            result.log_dir / "artifacts" / node.artifact_dir_name / "solution.py"
        ).read_text(encoding="utf-8") == node.code

    cfg, resumed = load_resume_state(
        run_id="manual",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )
    assert cfg.agent.mode == "legacy"
    assert cfg.manual_queue_only is True
    assert cfg.agent.steps == 2
    assert cfg.agent.search.code_ahead == 0
    assert next_generated_only_node(resumed, cfg=cfg) is resumed.nodes[0]
    assert (result.workspace_dir / "input").is_dir()
    assert (result.workspace_dir / "working").is_dir()


def test_seed_manual_legacy_appends_to_existing_run(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    desc_file = tmp_path / "task.md"
    desc_file.write_text("manual legacy task\n", encoding="utf-8")
    first = tmp_path / "01_first.py"
    second = tmp_path / "02_second.py"
    first.write_text("print('first')\n", encoding="utf-8")
    second.write_text("print('second')\n", encoding="utf-8")
    kwargs = {
        "data_dir": data_dir,
        "desc_file": desc_file,
        "logs_dir": tmp_path / "logs",
        "workspaces_dir": tmp_path / "workspaces",
        "prepare_workspace": False,
    }

    seed_manual_legacy_run(sources=(first,), **kwargs)
    result = seed_manual_legacy_run(sources=(second,), **kwargs)

    journal = serialize.load_json(result.log_dir / "journal.json", Journal)
    assert len(journal.nodes) == 2
    assert [node.step for node in journal.nodes] == [0, 1]
    assert all(node.status == "generated" for node in journal.nodes)


def test_seed_manual_legacy_rejects_non_python_source(tmp_path):
    source = tmp_path / "not_python.txt"
    source.write_text("text\n", encoding="utf-8")

    try:
        seed_manual_legacy_run(
            sources=(source,),
            logs_dir=tmp_path / "logs",
            workspaces_dir=tmp_path / "workspaces",
            prepare_workspace=False,
        )
    except ValueError as exc:
        assert ".py" in str(exc)
    else:
        raise AssertionError("non-Python source should be rejected")
