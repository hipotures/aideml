import datetime as dt
import logging
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import humanize
from .autogluon_preprocess import (
    BASELINE_PLAN_PREFIX,
    baseline_preprocess_source,
    build_autogluon_wrapper,
    extract_preprocess_source,
    infer_sample_submission_columns,
    is_autogluon_preprocess_mode,
    parse_result_marker,
    validate_preprocess_source,
)
from .backend import FunctionSpec, query
from .backend.utils import write_llm_response_code
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .research import format_research_hints_for_prompt, load_latest_research_hints
from .synthesis import SYNTHESIS_PLAN_PREFIX
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import (
    extract_code,
    extract_jsons,
    extract_text_up_to_code,
    wrap_code,
)

logger = logging.getLogger("aide")


ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": ["is_bug", "summary", "metric", "lower_is_better"],
    },
    description="Submit a review evaluating the output of the training script.",
)


def _parse_review_response(response: Any) -> dict[str, Any] | None:
    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        for parsed in extract_jsons(response):
            if isinstance(parsed, dict):
                return parsed

    return None


def _mark_invalid_review_response(node: Node, response: Any) -> None:
    node.analysis = (
        "Invalid review response from feedback model; marking this node as buggy "
        f"so the run can continue. Response type: {type(response).__name__}."
    )
    node.is_buggy = True
    node.metric = WorstMetricValue()


def _is_synthesis_node(node: Node) -> bool:
    return str(node.plan or "").startswith(SYNTHESIS_PLAN_PREFIX)


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.active_parent_node: Node | None = None
        self.active_node: Node | None = None
        self.active_stage: str | None = None
        self.active_stage_started_at: float | None = None
        self._pending_node_ctime: float | None = None
        self._pending_llm_log_dir: Path | None = None

    def set_active_stage(self, stage: str | None) -> None:
        self.active_stage = stage
        self.active_stage_started_at = time.monotonic() if stage is not None else None

    def _node_artifact_dir(self, node: Node) -> Path:
        timestamp = dt.datetime.fromtimestamp(node.ctime).strftime("%Y%m%dT%H%M%S")
        return Path(self.cfg.log_dir) / "artifacts" / timestamp

    def _new_node(
        self,
        *,
        plan: str,
        code: str,
        parent: Node | None = None,
    ) -> Node:
        kwargs: dict[str, Any] = {"plan": plan, "code": code, "parent": parent}
        if self._pending_node_ctime is not None:
            kwargs["ctime"] = self._pending_node_ctime
        return Node(**kwargs)

    def _generation_log_context(self) -> dict[str, Any]:
        parent = self.active_parent_node
        return {
            "phase": "generate",
            "run_id": self.cfg.exp_name,
            "parent_node_id": parent.id if parent is not None else None,
            "parent_stage": parent.stage_name if parent is not None else None,
            "agent_mode": self.acfg.mode,
            "node_ctime": self._pending_node_ctime,
        }

    def _review_log_context(self, node: Node) -> dict[str, Any]:
        return {
            "phase": "review",
            "run_id": self.cfg.exp_name,
            "node_id": node.id,
            "node_step": node.step,
            "node_stage": node.stage_name,
            "node_ctime": node.ctime,
            "agent_mode": self.acfg.mode,
        }

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (
                    n.is_leaf
                    and n.debug_depth < search_cfg.max_debug_depth
                    and not n.is_submission_contract_error
                )
            ]
            if debuggable_nodes:
                logger.debug("[search policy] debugging")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] not debugging by chance")

        # back to drafting if no nodes to improve
        good_nodes = [
            n
            for n in self.journal.good_nodes
            if not n.is_in_submission_contract_error_branch
        ]
        if not good_nodes:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return None

        synthesis_leaf_nodes = [
            n
            for n in good_nodes
            if n.is_leaf and _is_synthesis_node(n)
        ]
        if synthesis_leaf_nodes:
            logger.debug("[search policy] improving synthesis leaf")
            return max(synthesis_leaf_nodes, key=lambda n: n.metric)

        # greedy
        greedy_node = max(good_nodes, key=lambda n: n.metric)
        logger.debug("[search policy] greedy node selected")
        return greedy_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "catboost",
            "autogluon",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    @property
    def _prompt_autogluon_preprocess_guideline(self):
        return {
            "AutoGluon preprocess mode contract": [
                "You are writing only the feature preprocessing function for a fixed AutoGluon training wrapper.",
                "Return a single markdown code block containing exactly one top-level function: def preprocess(df: pd.DataFrame) -> pd.DataFrame.",
                "The df argument contains concatenated train features followed by Kaggle prediction/test features.",
                "The target column, id column, and train/test split marker are intentionally not present in df.",
                "No helper row-id column is exposed to preprocess(df); row order must be preserved.",
                "Do not create, reference, infer, or use target, id, `__is_train__`, or `__aide_row_id__` as features.",
                "Do not read files, write files, train models, create validation splits, save submissions, or call AutoGluon. The fixed wrapper does all of that.",
                "Do not change row count or reorder rows.",
                "Create deterministic, leakage-safe feature engineering only. Shared train+test operations like dtype cleanup, frequency encoding, and category normalization are allowed if they use only non-target columns.",
            ]
        }

    def _add_research_hints(self, prompt: dict[str, Any]) -> None:
        if not self.cfg.research.enabled:
            return
        hints = load_latest_research_hints(self.cfg.log_dir)
        if hints is not None:
            prompt["External research hints"] = format_research_hints_for_prompt(hints)

    def _autogluon_target_column(self) -> str | None:
        columns = infer_sample_submission_columns(self.cfg.workspace_dir / "input")
        return columns[1] if columns is not None else None

    def _add_autogluon_context(self, prompt: dict[str, Any]) -> None:
        target_col = self._autogluon_target_column()
        if target_col is not None:
            prompt["Fixed AutoGluon wrapper context"] = {
                "Target column": target_col,
                "Important": (
                    "The fixed wrapper removes this target before calling "
                    "preprocess(df), keeps y_train internally, and adds it back "
                    "only after preprocessing for AutoGluon training. It also "
                    "removes the id column and does not expose a train/test marker. "
                    "preprocess(df) must not reference or create target, id, or "
                    "split-marker columns."
                ),
            }

    def _wrap_autogluon_preprocess_node(
        self,
        *,
        plan: str,
        code: str,
        parent: Node | None = None,
    ) -> Node:
        try:
            preprocess_source = extract_preprocess_source(code)
            validate_preprocess_source(
                preprocess_source,
                target_col=self._autogluon_target_column(),
            )
            wrapped_code = build_autogluon_wrapper(preprocess_source, self.cfg)
        except ValueError as exc:
            wrapped_code = f"raise ValueError({str(exc)!r})\n"
        return self._new_node(plan=plan, code=wrapped_code, parent=parent)

    def _autogluon_raw_baseline(self) -> Node:
        code = build_autogluon_wrapper(baseline_preprocess_source(), self.cfg)
        return self._new_node(
            plan=(
                f"{BASELINE_PLAN_PREFIX}: raw features with the configured "
                "fixed AutoGluon runner."
            ),
            code=code,
        )

    def _previous_preprocess_source(self, parent_node: Node) -> str:
        try:
            return extract_preprocess_source(parent_node.code)
        except ValueError:
            return parent_node.code

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                reasoning_effort=self.acfg.code.reasoning_effort,
                temperature=self.acfg.code.temp,
                llm_log_dir=self._pending_llm_log_dir,
                llm_log_context=self._generation_log_context(),
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                write_llm_response_code(
                    log_dir=self._pending_llm_log_dir,
                    code=code,
                )
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self) -> Node:
        if is_autogluon_preprocess_mode(self.cfg):
            return self._draft_autogluon_preprocess()

        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        self._add_research_hints(prompt)
        plan, code = self.plan_and_code_query(prompt)
        return self._new_node(plan=plan, code=code)

    def _draft_autogluon_preprocess(self) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "A fixed AutoGluon runner will handle model training, validation, "
                "and submission generation. Your job is to design leakage-safe "
                "feature preprocessing for that runner."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Preprocessing sketch guideline": [
                "The solution sketch should be 3-5 sentences describing the feature engineering idea.",
                "Keep the first solution relatively simple and deterministic.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_autogluon_preprocess_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        self._add_autogluon_context(prompt)
        self._add_research_hints(prompt)
        plan, code = self.plan_and_code_query(prompt)
        return self._wrap_autogluon_preprocess_node(plan=plan, code=code)

    def _improve(self, parent_node: Node) -> Node:
        if is_autogluon_preprocess_mode(self.cfg):
            return self._improve_autogluon_preprocess(parent_node)

        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        self._add_research_hints(prompt)
        plan, code = self.plan_and_code_query(prompt)
        return self._new_node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def _improve_autogluon_preprocess(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are improving only the feature preprocessing function for a "
                "fixed AutoGluon training wrapper. Keep the wrapper behavior "
                "unchanged and make one atomic, leakage-safe feature improvement."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Previous preprocess function": wrap_code(
                self._previous_preprocess_source(parent_node)
            ),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Preprocessing improvement sketch guideline": [
                "The solution sketch should describe one specific feature engineering improvement.",
                "Make the change atomic so the AutoGluon wrapper can evaluate its effect.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_autogluon_preprocess_guideline
        self._add_autogluon_context(prompt)
        self._add_research_hints(prompt)
        plan, code = self.plan_and_code_query(prompt)
        return self._wrap_autogluon_preprocess_node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def _debug(self, parent_node: Node) -> Node:
        if is_autogluon_preprocess_mode(self.cfg):
            return self._debug_autogluon_preprocess(parent_node)

        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        self._add_autogluon_context(prompt)
        self._add_research_hints(prompt)
        plan, code = self.plan_and_code_query(prompt)
        return self._new_node(plan=plan, code=code, parent=parent_node)

    def _debug_autogluon_preprocess(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are fixing a buggy feature preprocessing function used by a "
                "fixed AutoGluon training wrapper. Revise only preprocess(df); "
                "do not write a full model pipeline."
            ),
            "Task description": self.task_desc,
            "Previous preprocess function": wrap_code(
                self._previous_preprocess_source(parent_node)
            ),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix preprocessing sketch guideline": [
                "Describe the cause of the preprocessing failure and the narrow fix.",
                "Keep the function deterministic and leakage-safe.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_autogluon_preprocess_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        self._add_research_hints(prompt)
        plan, code = self.plan_and_code_query(prompt)
        return self._wrap_autogluon_preprocess_node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def prepare_step(self) -> Node | None:
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        self.active_parent_node = parent_node
        return parent_node

    def generate_node(
        self,
        parent_node: Node | None,
        *,
        node_ctime: float | None = None,
        llm_log_dir: Path | None = None,
    ) -> Node:
        self.set_active_stage("generating")
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")
        previous_ctime = self._pending_node_ctime
        previous_log_dir = self._pending_llm_log_dir
        self._pending_node_ctime = node_ctime
        self._pending_llm_log_dir = llm_log_dir

        try:
            if (
                parent_node is None
                and is_autogluon_preprocess_mode(self.cfg)
                and not self.journal.nodes
            ):
                return self._autogluon_raw_baseline()
            if parent_node is None:
                return self._draft()
            if parent_node.is_buggy:
                return self._debug(parent_node)
            return self._improve(parent_node)
        finally:
            self._pending_node_ctime = previous_ctime
            self._pending_llm_log_dir = previous_log_dir

    def execute_node(
        self, node: Node, exec_callback: ExecCallbackType
    ) -> ExecutionResult:
        self.active_node = node
        self.set_active_stage("executing")
        return exec_callback(node.code, True)

    def review_node(self, node: Node, exec_result: ExecutionResult) -> None:
        self.set_active_stage("reviewing")
        self.parse_exec_result(
            node=node,
            exec_result=exec_result,
        )

    def clear_active_step(self) -> None:
        self.active_parent_node = None
        self.active_node = None
        self.set_active_stage(None)

    def step(self, exec_callback: ExecCallbackType):
        parent_node = self.prepare_step()

        try:
            result_node = self.generate_node(parent_node)
            exec_result = self.execute_node(result_node, exec_callback)
            self.review_node(result_node, exec_result)
            self.journal.append(result_node)
        finally:
            self.clear_active_step()

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)
        marker_response = parse_result_marker(node.term_out)
        if marker_response is not None:
            metric = marker_response.get("metric")
            if not isinstance(metric, (float, int)) or isinstance(metric, bool):
                metric = None
            node.analysis = str(marker_response.get("summary", ""))
            node.is_buggy = (
                bool(marker_response.get("is_bug"))
                or node.exc_type is not None
                or metric is None
            )
            if node.is_buggy:
                node.metric = WorstMetricValue()
            else:
                node.metric = MetricValue(
                    metric,
                    maximize=not bool(marker_response.get("lower_is_better")),
                )
            return

        prompt = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            ),
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        response = query(
            system_message=prompt,
            user_message=None,
            func_spec=review_func_spec,
            model=self.acfg.feedback.model,
            reasoning_effort=self.acfg.feedback.reasoning_effort,
            temperature=self.acfg.feedback.temp,
            llm_log_dir=self._node_artifact_dir(node),
            llm_log_prefix="review",
            llm_log_context=self._review_log_context(node),
        )
        parsed_response = _parse_review_response(response)
        if parsed_response is None:
            _mark_invalid_review_response(node, response)
            return

        required_keys = {"is_bug", "summary", "metric", "lower_is_better"}
        if not required_keys.issubset(parsed_response):
            _mark_invalid_review_response(node, response)
            return

        # if the metric isn't a float then fill the metric with the worst metric
        metric = parsed_response["metric"]
        if not isinstance(metric, (float, int)) or isinstance(metric, bool):
            metric = None

        node.analysis = str(parsed_response["summary"])
        node.is_buggy = (
            bool(parsed_response["is_bug"])
            or node.exc_type is not None
            or metric is None
        )

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                metric, maximize=not bool(parsed_response["lower_is_better"])
            )
