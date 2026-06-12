from __future__ import annotations

from scripts.reset_buggy_root_hypotheses import apply_resets, plan_resets


def test_reset_buggy_root_hypotheses_removes_root_subtree() -> None:
    journal = {
        "nodes": [
            {
                "id": "bug-root",
                "step": 1,
                "status": "bug",
                "is_buggy": True,
                "parent": None,
                "research_mode": "hypothesis",
                "research_hypotheses_offered": ["000010"],
            },
            {
                "id": "bug-child",
                "step": 2,
                "status": "generated",
                "is_buggy": False,
                "parent": "bug-root",
                "research_mode": "hypothesis",
                "research_hypotheses_offered": ["000010"],
            },
            {
                "id": "ok-root",
                "step": 3,
                "status": "ok",
                "is_buggy": False,
                "parent": None,
                "research_mode": "hypothesis",
                "research_hypotheses_offered": ["000011"],
            },
        ],
        "node2parent": {"bug-child": "bug-root"},
    }

    resets = plan_resets(journal)
    apply_resets(journal, resets)

    assert [reset.hypothesis_id for reset in resets] == ["000010"]
    assert resets[0].removed_node_count == 2
    assert [node["id"] for node in journal["nodes"]] == ["ok-root"]
    assert journal["node2parent"] == {}


def test_reset_buggy_root_hypotheses_keeps_non_root_bug() -> None:
    journal = {
        "nodes": [
            {
                "id": "ok-root",
                "status": "ok",
                "is_buggy": False,
                "parent": None,
                "research_mode": "hypothesis",
                "research_hypotheses_offered": ["000010"],
            },
            {
                "id": "bug-child",
                "status": "bug",
                "is_buggy": True,
                "parent": "ok-root",
                "research_mode": "hypothesis",
                "research_hypotheses_offered": ["000010"],
            },
        ],
        "node2parent": {"bug-child": "ok-root"},
    }

    resets = plan_resets(journal)
    apply_resets(journal, resets)

    assert resets == []
    assert [node["id"] for node in journal["nodes"]] == ["ok-root", "bug-child"]
    assert journal["node2parent"] == {"bug-child": "ok-root"}
