import json

from aide.journal import Journal
from aide.agent import Agent
from aide.refactor_sidecar import (
    RefactorConfig,
    extract_python_code,
    maybe_refactor_response_py,
    validate_refactored_code,
)
from aide.utils.config import _load_cfg, prep_cfg


VALID_REFACTORED_CODE = """\
from aide_refactor_runtime import aide_stage, finalize_aide_artifacts


def main():
    with aide_stage("setup_stage"):
        print("refactored")


if __name__ == "__main__":
    try:
        main()
    finally:
        finalize_aide_artifacts()
"""


def test_extract_one_python_block():
    code, status = extract_python_code("```python\nimport os\nprint('x')\n```")
    assert status == "ok_fenced"
    assert "print" in code


def test_extract_raw_code():
    code, status = extract_python_code("import os\n\ndef main():\n    pass\n")
    assert status == "ok_raw_code"
    assert "def main" in code


def test_extract_multiple_blocks_fails():
    raw = "```python\nprint(1)\n```\n```python\nprint(2)\n```"
    code, status = extract_python_code(raw)
    assert code is None
    assert status == "multiple_python_blocks"


def test_validate_refactored_code_accepts_minimal_contract():
    valid, status = validate_refactored_code(VALID_REFACTORED_CODE)

    assert valid is True
    assert status == "ok"


def test_validate_refactored_code_requires_stage_and_finalize():
    valid, status = validate_refactored_code(
        "from aide_refactor_runtime import aide_stage\n"
        "with aide_stage('setup_stage'):\n"
        "    print('x')\n"
    )

    assert valid is False
    assert status == "missing_finalize_aide_artifacts"


def _write_refactor_inputs(tmp_path):
    prompt = tmp_path / "prompt.md"
    contract = tmp_path / "contract.md"
    api = tmp_path / "api.md"
    runtime = tmp_path / "aide_refactor_runtime.py"
    prompt.write_text(
        "{{EXECUTION_CONTRACT}}\n{{AIDE_REFACTOR_RUNTIME_API}}\n{{RESPONSE_PY}}",
        encoding="utf-8",
    )
    contract.write_text("contract", encoding="utf-8")
    api.write_text("api", encoding="utf-8")
    runtime.write_text("# runtime\n", encoding="utf-8")
    return prompt, contract, api, runtime


def test_refactor_disabled(tmp_path):
    response = tmp_path / "response.py"
    response.write_text("print('original')\n", encoding="utf-8")

    called = {"n": 0}

    def call_model(prompt, model, timeout_s):
        called["n"] += 1
        return "```python\nprint('refactored')\n```"

    meta = maybe_refactor_response_py(
        response_py_path=response,
        artifact_dir=tmp_path,
        call_model=call_model,
        config=RefactorConfig(enabled=False),
    )

    assert meta["status"] == "disabled"
    assert called["n"] == 0
    assert not (tmp_path / "response_refactored.py").exists()


def test_refactor_success(tmp_path):
    response = tmp_path / "response.py"
    response.write_text("print('original')\n", encoding="utf-8")

    prompt, contract, api, runtime = _write_refactor_inputs(tmp_path)

    def call_model(prompt_text, model, timeout_s):
        assert "print('original')" in prompt_text
        return f"```python\n{VALID_REFACTORED_CODE}```"

    meta = maybe_refactor_response_py(
        response_py_path=response,
        artifact_dir=tmp_path,
        call_model=call_model,
        config=RefactorConfig(
            enabled=True,
            model="mock",
            prompt_path=prompt,
            execution_contract_path=contract,
            runtime_api_path=api,
            runtime_source_path=runtime,
        ),
    )

    assert meta["status"] == "success"
    assert "refactored" in (tmp_path / "response_refactored.py").read_text(encoding="utf-8")
    assert (tmp_path / "aide_refactor_runtime.py").read_text(encoding="utf-8") == "# runtime\n"
    assert not (tmp_path / "response_refactor_request.md").exists()
    assert not (tmp_path / "response_refactor_raw.md").exists()
    assert response.read_text(encoding="utf-8") == "print('original')\n"


def test_refactor_validation_failed(tmp_path):
    response = tmp_path / "response.py"
    response.write_text("print('original')\n", encoding="utf-8")
    prompt, contract, api, runtime = _write_refactor_inputs(tmp_path)

    def call_model(prompt_text, model, timeout_s):
        return "```python\nprint('still free style')\n```"

    meta = maybe_refactor_response_py(
        response_py_path=response,
        artifact_dir=tmp_path,
        call_model=call_model,
        config=RefactorConfig(
            enabled=True,
            model="mock",
            prompt_path=prompt,
            execution_contract_path=contract,
            runtime_api_path=api,
            runtime_source_path=runtime,
        ),
    )

    assert meta["status"] == "validation_failed"
    assert meta["error_type"] == "missing_runtime_import"
    assert not (tmp_path / "response_refactored.py").exists()


def test_refactor_parse_failed(tmp_path):
    response = tmp_path / "response.py"
    response.write_text("print('original')\n", encoding="utf-8")

    prompt, contract, api, runtime = _write_refactor_inputs(tmp_path)

    def call_model(prompt_text, model, timeout_s):
        return "I cannot do this."

    meta = maybe_refactor_response_py(
        response_py_path=response,
        artifact_dir=tmp_path,
        call_model=call_model,
        config=RefactorConfig(
            enabled=True,
            model="mock",
            prompt_path=prompt,
            execution_contract_path=contract,
            runtime_api_path=api,
            runtime_source_path=runtime,
        ),
    )

    assert meta["status"] == "parse_failed"
    assert not (tmp_path / "response_refactored.py").exists()


def test_refactor_skips_input_too_large(tmp_path):
    response = tmp_path / "response.py"
    response.write_text("print('original')\n", encoding="utf-8")
    prompt, contract, api, runtime = _write_refactor_inputs(tmp_path)

    called = {"n": 0}

    def call_model(prompt_text, model, timeout_s):
        called["n"] += 1
        return f"```python\n{VALID_REFACTORED_CODE}```"

    meta = maybe_refactor_response_py(
        response_py_path=response,
        artifact_dir=tmp_path,
        call_model=call_model,
        config=RefactorConfig(
            enabled=True,
            model="mock",
            prompt_path=prompt,
            execution_contract_path=contract,
            runtime_api_path=api,
            runtime_source_path=runtime,
            max_input_chars=10,
        ),
    )

    assert meta["status"] == "skipped_input_too_large"
    assert called["n"] == 0
    assert not (tmp_path / "response_refactored.py").exists()


def test_agent_refactor_pass_writes_prefixed_llm_artifacts_without_replacing_code(
    tmp_path,
    monkeypatch,
):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "Predict a target"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "refactor-test"
    cfg.refactor.enabled = True
    cfg.refactor.model = "mock-refactor-model"
    cfg = prep_cfg(cfg)

    monkeypatch.delenv("AIDE_REFACTOR_ENABLED", raising=False)
    monkeypatch.delenv("AIDE_REFACTOR_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setitem(Agent.plan_and_code_query.__globals__["query"].__globals__, "_llm_call_counter", 0)
    monkeypatch.setattr("aide.backend.determine_provider", lambda model: "openai")

    def fake_query_func(**kwargs):
        if "Refactor `response.py`" in str(kwargs.get("system_message")):
            assert kwargs["model"] == cfg.refactor.model
            output = f"```python\n{VALID_REFACTORED_CODE}```"
        else:
            output = "I will keep this simple.\n\n```python\nprint('original')\n```"
        return output, 0.1, 1, 1, {"model": kwargs["model"]}

    monkeypatch.setitem(
        Agent.plan_and_code_query.__globals__["query"].__globals__["provider_to_query_func"],
        "openai",
        fake_query_func,
    )

    artifact_dir = tmp_path / "artifact"
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    agent._pending_llm_log_dir = artifact_dir

    plan, code = agent.plan_and_code_query({"Instructions": "write code"})

    assert plan.startswith("I will keep")
    assert "original" in code
    assert (artifact_dir / "response.py").read_text(encoding="utf-8") == code
    assert "refactored" in (artifact_dir / "response_refactored.py").read_text(encoding="utf-8")
    assert (artifact_dir / "aide_refactor_runtime.py").exists()
    assert (artifact_dir / "refactor_request.md").exists()
    assert (artifact_dir / "refactor_response_raw.txt").exists()
    assert (artifact_dir / "refactor_status.json").exists()
    assert (artifact_dir / "refactor_stderr.log").exists()
    assert not (artifact_dir / "refactor_process_stdout.log").exists()
    assert not (artifact_dir / "refactor_solution.py").exists()
    meta = json.loads((artifact_dir / "response_refactor_meta.json").read_text())
    assert meta["status"] == "success"
